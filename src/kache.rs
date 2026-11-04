// K2Rmini with kache-hash as its k-mer index: the patterns' exact k-mers, as they occur, in a concurrent table built by
// all threads, each canonical k-mer with the mask of the orientations it occurs in; and counted per read for pass 2,
// in one call per message. The minimizer index is built concurrently too,
// in shards. Everything else is 1.0.0's: the parser, pass 1, the thresholds, and the output.

use crate::index::{self, ShardedSet, LOSSY};
use crate::{mem_usage_gb, Args, Threshold, CONFIG_FILTER, MATCH_N, MSG_LEN_THRESHOLD};

use crossbeam_channel::bounded;
use helicase::input::*;
use helicase::*;
use kache_hash_sys::{Builder, Op, Options, Reduction, Table};
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::{PackedSeqVec, Seq, SeqVec};

use core::mem::swap;
use std::fs::File;
use std::io::{self, BufWriter, Write, stdout};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

const KMER_LEN: usize = 31; // The k-mer length kache-hash supports.
const INIT_KEYS: u64 = 1 << 22; // The table's initial room; it grows as the patterns need.
const LOAD_FACTOR: f64 = 0.8;
// The bits of a pattern k-mer's mask in the table: it occurs as is, i.e. as its canonical form, and as that form's
// reverse complement.
const FWD: u8 = 1;
const REV: u8 = 2;

pub fn run(args: &Args) -> io::Result<()> {
    if args.k != KMER_LEN || !kache_hash_sys::supports::<u8>(args.k) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--index kache supports k = 31 only"));
    }
    let threads = args.threads.unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

    eprintln!("Indexing k-mers and minimizers of interest...");
    let start = Instant::now();
    let options = Options {
        k: args.k,
        expected_keys: INIT_KEYS,
        load_factor: LOAD_FACTOR,
        resize_workers: threads,
        pow2_capacity: true,
    };
    let builder = Builder::<u8>::new(&options).map_err(io::Error::other)?;
    let (min_dict, t) = index::ingest(&args.patterns, args.k, args.m, threads, || {
        let mut ins = builder.inserter().expect("Failed to register with kache-hash");
        move |seqs: &[&[u8]]| {
            ins.insert_oriented(seqs, Op::Or, FWD, REV).expect("Failed to insert patterns into kache-hash");
        }
    })?;
    let kmer_dict = builder.freeze();
    let t_size = Instant::now();
    let kmer_c = kmer_dict.reduce(Reduction::Popcount, FWD | REV, threads).map_err(io::Error::other)? as usize;
    let t_size = t_size.elapsed();
    eprintln!("Took {:.02} s, RAM: {:.03} GB", start.elapsed().as_secs_f64(), mem_usage_gb());
    eprintln!("Indexed {} k-mers and {} minimizers.", kmer_c, min_dict.len());
    let s = kmer_dict.stats().map_err(io::Error::other)?;
    eprintln!(
        "Index parts: scan {:.02} s; threads' TM {:.02} s and TQ {:.02} s (sums over {} threads); TM freeze {:.02} s; \
         size {:.02} s. kache-hash: {} keys, {} slots, {:.03} GB of buckets, placed {:.02}/{:.02}/{:.02}%.",
        t.scan.as_secs_f64(),
        t.tm.as_secs_f64(),
        t.tq.as_secs_f64(),
        threads,
        t.freeze.as_secs_f64(),
        t_size.as_secs_f64(),
        s.keys,
        s.capacity,
        s.bytes as f64 / 1e9,
        s.primary_pct,
        s.secondary_pct,
        s.overflow_pct
    );

    eprintln!("Filtering sequences in parallel...");
    let start = Instant::now();
    filter(args, threads, &kmer_dict, kmer_c, &min_dict)?;
    eprintln!("Took {:.02} s, RAM: {:.03} GB", start.elapsed().as_secs_f64(), mem_usage_gb());

    Ok(())
}

// A read that pass 1 keeps, for pass 2: the bounds of its header and sequence in the message, its threshold, and what
// kache-hash counts the k-mers of, in the message (`joined` false) or in the consumer's buffer of joined reads.
struct Candidate {
    id: (usize, usize),
    seq: (usize, usize),
    threshold: usize,
    view: (usize, usize),
    joined: bool,
}

// Returns the bounds of what pass 2 counts the k-mers of, for the read at `seq_bounds` of `seqs`: itself, if it has
// ACGT bases only; else the concatenation of its pieces of at least `k` bases between runs of N, with its other bytes
// mapped as 1.0.0 packs them, into `joined`, so that the k-mers are 1.0.0's, junctions included.
fn view(seqs: &[u8], seq_bounds: (usize, usize), k: usize, joined: &mut Vec<u8>) -> ((usize, usize), bool) {
    let seq = &seqs[seq_bounds.0..seq_bounds.1];
    if index::is_acgt(seq) {
        return (seq_bounds, false);
    }

    let b = joined.len();
    for piece in MATCH_N.split(seq).filter(|p| p.len() >= k) {
        joined.extend(piece.iter().map(|&c| LOSSY[c as usize]));
    }
    ((b, joined.len()), true)
}

// Filters the reads as 1.0.0's `process_query_streaming` does, with the k-mer index `kmer_dict` of `kmer_c` k-mers and
// the minimizer index `min_dict`.
fn filter(args: &Args, num_consumers: usize, kmer_dict: &Table<u8>, kmer_c: usize, min_dict: &ShardedSet) -> io::Result<()> {
    let kmer_size: usize = args.k;
    let minimizer_size: usize = args.m;
    let window_size: usize = kmer_size - minimizer_size + 1;
    let threshold = args.threshold;
    let output = args.output.clone();

    let path = args.file.clone();
    let (record_tx, record_rx) = bounded(2 * num_consumers);
    let (result_tx, result_rx) = bounded::<(Vec<u8>, Vec<u8>)>(4 * num_consumers);

    // Each kept read's header, length, threshold and count, for checks.
    let dump = std::env::var("K2RMINI_DUMP_COUNTS").ok().map(|p| (p, Mutex::new(Vec::<u8>::new())));

    thread::scope(|scope| -> io::Result<()> {
        let producer_handle = scope.spawn(move || {
            let mut ids = Vec::new();
            let mut seqs = Vec::new();
            let mut ends = Vec::new();
            let mut parser =
                FastxParser::<CONFIG_FILTER>::from_file(&path).expect("Failed to parse file to filter");
            while let Some(_) = parser.next() {
                let id = parser.get_header();
                let seq = parser.get_dna_string();
                if seq.len() < kmer_size {
                    continue;
                }
                ids.extend_from_slice(id);
                seqs.extend_from_slice(seq);
                ends.push((ids.len(), seqs.len()));
                if seqs.len() >= MSG_LEN_THRESHOLD {
                    let mut tmp_ids = Vec::new();
                    let mut tmp_seqs = Vec::new();
                    let mut tmp_ends = Vec::new();
                    swap(&mut ids, &mut tmp_ids);
                    swap(&mut seqs, &mut tmp_seqs);
                    swap(&mut ends, &mut tmp_ends);
                    if record_tx.send((tmp_ids, tmp_seqs, tmp_ends)).is_err() {
                        break;
                    }
                }
            }
            if !seqs.is_empty() {
                record_tx.send((ids, seqs, ends)).unwrap();
            }
        });

        let mut consumer_handles = Vec::with_capacity(num_consumers);
        for _ in 0..num_consumers {
            let record_rx_clone = record_rx.clone();
            let result_tx_clone = result_tx.clone();
            let dump = &dump;

            let handle = scope.spawn(move || {
                let mini_builder = minimizers(minimizer_size, window_size);
                let mut sk_pos = Vec::new();
                let mut mini_pos = Vec::new();
                let mut cand: Vec<Candidate> = Vec::new();
                let mut joined = Vec::new();
                let mut counts = Vec::new();
                let mut dumped = Vec::new();
                while let Ok((ids, seqs, ends)) = record_rx_clone.recv() {
                    cand.clear();
                    joined.clear();
                    if ends.len() == 1 {
                        // a single long seq
                        let seq = &seqs;

                        let kmer_threshold: usize = match threshold {
                            Threshold::Absolute(n) => n,
                            Threshold::Relative(f) => {
                                (((seq.len().saturating_sub(kmer_size) + 1) as f64) * f).ceil() as usize
                            }
                        }
                        .min(kmer_c);
                        let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

                        let mut packed_seq = PackedSeqVec::default();
                        MATCH_N
                            .split(seq)
                            .filter(|&seq| seq.len() >= kmer_size)
                            .for_each(|seq| {
                                packed_seq.push_ascii(seq);
                            });

                        mini_pos.clear();
                        mini_builder.run(packed_seq.as_slice(), &mut mini_pos);
                        let shared_min_count = mini_pos
                            .iter()
                            .copied()
                            .map(|pos| {
                                packed_seq
                                    .slice((pos as usize)..(pos as usize + minimizer_size))
                                    .as_u64()
                            })
                            .filter(|word| min_dict.contains(word))
                            .count();

                        if shared_min_count < minimizer_threshold {
                            continue;
                        }

                        let (v, j) = view(&seqs, (0, seqs.len()), kmer_size, &mut joined);
                        cand.push(Candidate { id: (0, ids.len()), seq: (0, seqs.len()), threshold: kmer_threshold, view: v, joined: j });
                    } else {
                        // multiple short seqs
                        let mut packed_seqs = PackedSeqVec::default();
                        let mut packed_ends = Vec::with_capacity(ends.len());
                        let mut seq_start = 0;
                        for (_, seq_end) in ends.iter().copied() {
                            let seq = &seqs[seq_start..seq_end];
                            MATCH_N
                                .split(seq)
                                .filter(|&seq| seq.len() >= kmer_size)
                                .for_each(|seq| {
                                    packed_seqs.push_ascii(seq);
                                });
                            packed_ends.push(packed_seqs.len());
                            seq_start = seq_end;
                        }

                        sk_pos.clear();
                        mini_pos.clear();
                        mini_builder
                            .super_kmers(&mut sk_pos)
                            .run(packed_seqs.as_slice(), &mut mini_pos);

                        let mut id_start = 0;
                        let mut seq_start = 0;
                        let mut mini_idx = 0;
                        for ((id_end, seq_end), packed_end) in ends.iter().copied().zip(packed_ends) {
                            let seq = &seqs[seq_start..seq_end];
                            let kmer_last = packed_end - kmer_size + 1;

                            let kmer_threshold: usize = match threshold {
                                Threshold::Absolute(n) => n,
                                Threshold::Relative(f) => {
                                    (((seq.len().saturating_sub(kmer_size) + 1) as f64) * f).ceil()
                                        as usize
                                }
                            }
                            .min(kmer_c);
                            let minimizer_threshold: usize = kmer_threshold.div_ceil(window_size);

                            let mut shared_min_count = 0;
                            while mini_idx < sk_pos.len() && sk_pos[mini_idx] < kmer_last as u32 {
                                let pos = mini_pos[mini_idx] as usize;
                                let word = packed_seqs.slice(pos..(pos + minimizer_size)).as_u64();
                                shared_min_count += if min_dict.contains(&word) { 1 } else { 0 };
                                mini_idx += 1;
                            }
                            while mini_idx + 1 < sk_pos.len()
                                && sk_pos[mini_idx + 1] <= packed_end as u32
                            {
                                mini_idx += 1;
                            }

                            if shared_min_count >= minimizer_threshold {
                                let (v, j) = view(&seqs, (seq_start, seq_end), kmer_size, &mut joined);
                                cand.push(Candidate {
                                    id: (id_start, id_end),
                                    seq: (seq_start, seq_end),
                                    threshold: kmer_threshold,
                                    view: v,
                                    joined: j,
                                });
                            }

                            id_start = id_end;
                            seq_start = seq_end;
                        }
                    }

                    if cand.is_empty() {
                        continue;
                    }

                    // Pass 2: the candidates' k-mers, counted in kache-hash at once.
                    let views: Vec<&[u8]> = cand
                        .iter()
                        .map(|c| if c.joined { &joined[c.view.0..c.view.1] } else { &seqs[c.view.0..c.view.1] })
                        .collect();
                    counts.clear();
                    counts.resize(views.len(), 0u64);
                    kmer_dict.count_masked(&views, FWD, REV, &mut counts).expect("Failed to count k-mers in kache-hash");

                    for (c, &n) in cand.iter().zip(&counts) {
                        if dump.is_some() {
                            dumped.extend_from_slice(&ids[c.id.0..c.id.1]);
                            let _ = writeln!(dumped, "\t{}\t{}\t{}", c.seq.1 - c.seq.0, c.threshold, n);
                        }
                        if n as usize >= c.threshold {
                            let _ = result_tx_clone.send((ids[c.id.0..c.id.1].to_vec(), seqs[c.seq.0..c.seq.1].to_vec()));
                        }
                    }
                }

                if let Some((_, d)) = dump {
                    d.lock().unwrap().extend_from_slice(&dumped);
                }
            });

            consumer_handles.push(handle);
        }

        drop(result_tx);

        let printer_handle = scope.spawn(move || {
            if let Some(out) = output {
                let file = File::create(out).expect("Failed to open output file");
                let mut writer = BufWriter::new(file);
                for (id, seq) in result_rx.iter() {
                    writer.write_all(b">")?;
                    writer.write_all(&id)?;
                    writer.write_all(b"\n")?;
                    writer.write_all(&seq)?;
                    writer.write_all(b"\n")?;
                }
            } else {
                for (id, seq) in result_rx.iter() {
                    stdout().write_all(b">")?;
                    stdout().write_all(&id)?;
                    stdout().write_all(b"\n")?;
                    stdout().write_all(&seq)?;
                    stdout().write_all(b"\n")?;
                }
            }
            io::Result::Ok(())
        });

        producer_handle.join().expect("Producer thread panicked");
        for handle in consumer_handles {
            handle.join().expect("Consumer thread panicked");
        }
        let _ = printer_handle.join().expect("Printer thread panicked");
        Ok(())
    })?;

    if let Some((p, d)) = dump {
        std::fs::write(p, d.into_inner().unwrap())?;
    }

    Ok(())
}
