// Parallel indexing of the pattern file: its records are cut into work items, which threads claim and describe. Each
// thread builds its share of the minimizer index, split into shards, and hands its items' sequences to a k-mer index.
// The k-mers and minimizers are those 1.0.0's sequential index takes, as helicase parses the file: line feeds are
// dropped, any other byte is a base, and non-ACGT bytes encode lossily.

use memchr::{memchr, memchr_iter};
use rustc_hash::FxHashSet;
use simd_minimizers::minimizers;
use simd_minimizers::packed_seq::{PackedSeqVec, Seq, SeqVec};

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

const ITEM_BYTES: usize = 4 << 20; // Bytes of sequence per work item: records longer than this are cut into runs.
const SHARD_BITS: u32 = 10; // The minimizer index has `2^SHARD_BITS` shards.
const FLUSH: usize = 512; // Values a thread buffers per shard before inserting them, under the shard's lock.

// Returns the shard of the minimizer `x`: the top bits of a multiplicative hash other than the shards' own, so that
// their keys keep their spread over hashbrown's control bits.
#[inline(always)]
fn shard_of(x: u64) -> usize {
    (x.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - SHARD_BITS)) as usize
}

// The minimizer index, in shards.
pub struct ShardedSet {
    shards: Vec<FxHashSet<u64>>,
}

impl ShardedSet {
    #[inline(always)]
    pub fn contains(&self, x: &u64) -> bool {
        // SAFETY: `shard_of` is below `2^SHARD_BITS`, the number of shards.
        unsafe { self.shards.get_unchecked(shard_of(*x)) }.contains(x)
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }
}

// A thread's buffers into the shards of a minimizer index being built.
struct ShardBuffers<'a> {
    shards: &'a [Mutex<FxHashSet<u64>>],
    buf: Vec<Vec<u64>>,
}

impl<'a> ShardBuffers<'a> {
    fn new(shards: &'a [Mutex<FxHashSet<u64>>]) -> Self {
        Self {
            shards,
            buf: (0..shards.len()).map(|_| Vec::with_capacity(FLUSH)).collect(),
        }
    }

    #[inline(always)]
    fn push(&mut self, x: u64) {
        let s = shard_of(x);
        self.buf[s].push(x);
        if self.buf[s].len() == FLUSH {
            self.flush(s);
        }
    }

    fn flush(&mut self, s: usize) {
        self.shards[s].lock().unwrap().extend(self.buf[s].drain(..));
    }
}

impl Drop for ShardBuffers<'_> {
    fn drop(&mut self) {
        for s in 0..self.buf.len() {
            self.flush(s);
        }
    }
}

// The pattern file, mapped read-only.
struct Mapped {
    ptr: *const u8,
    len: usize,
}

// SAFETY: the mapping is read-only.
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

impl Mapped {
    fn open(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Self { ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0 });
        }

        // SAFETY: a read-only private mapping of the whole file, which outlives the descriptor.
        let ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0)
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self { ptr: ptr as *const u8, len })
    }
}

impl Deref for Mapped {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the mapping holds `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: the mapping is unmapped once.
            unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        }
    }
}

// A run of a record's sequence: the bytes `[start, end)` of the file, line feeds included, then iff `ext`, its next
// `k - 1` bases up to `rec_end`, the record's end, which the next run starts with; so that every k-mer is in one run.
#[derive(Clone, Copy)]
struct Run {
    start: usize,
    end: usize,
    rec_end: usize,
    ext: bool,
}

// Returns the bounds of the sequences of the records of the FASTA or FASTQ file `buf`.
fn records(buf: &[u8], threads: usize) -> io::Result<Vec<(usize, usize)>> {
    let invalid = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, format!("patterns: {msg}"));
    match buf.first() {
        None => Ok(Vec::new()),
        Some(b'>') => {
            // The record starts, `>` at line starts, found in parallel: sequences hold no `>`.
            let chunk = buf.len().div_ceil(threads.max(1)).max(1);
            let mut starts: Vec<usize> = thread::scope(|s| {
                let parts: Vec<_> = (0..buf.len())
                    .step_by(chunk)
                    .map(|b| {
                        s.spawn(move || {
                            let e = (b + chunk).min(buf.len());
                            memchr_iter(b'>', &buf[b..e])
                                .map(|i| b + i)
                                .filter(|&p| p == 0 || buf[p - 1] == b'\n')
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                parts.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
            starts.push(buf.len());

            Ok(starts
                .windows(2)
                .map(|w| {
                    let seq = memchr(b'\n', &buf[w[0]..w[1]]).map_or(w[1], |i| w[0] + i + 1);
                    (seq, w[1])
                })
                .collect())
        }
        Some(b'@') => {
            // Four-line records: the header, the sequence, the separator, and the qualities.
            let mut recs = Vec::new();
            let mut p = 0;
            let line_end = |p: usize| memchr(b'\n', &buf[p..]).map_or(buf.len(), |i| p + i);
            while p < buf.len() {
                if buf[p] != b'@' {
                    return Err(invalid("malformed FASTQ record"));
                }
                let seq = line_end(p) + 1;
                let seq_end = line_end(seq.min(buf.len()));
                let sep = seq_end + 1;
                let qual = line_end(sep.min(buf.len())) + 1;
                recs.push((seq.min(buf.len()), seq_end));
                p = (line_end(qual.min(buf.len())) + 1).min(buf.len());
            }
            Ok(recs)
        }
        Some(_) => Err(invalid("not FASTA or FASTQ")),
    }
}

// Appends the bytes of `src` to `dst`, dropping its line feeds.
fn push_stripped(dst: &mut Vec<u8>, src: &[u8]) {
    let mut s = src;
    while let Some(i) = memchr(b'\n', s) {
        dst.extend_from_slice(&s[..i]);
        s = &s[i + 1..];
    }
    dst.extend_from_slice(s);
}

// Maps each byte to the base 1.0.0 indexes it as: ACGT bases, in either case, stay; any other byte becomes the base of
// its 2-bit code in packed-seq's lossy encoding, `(b >> 1) & 3` (N to G).
pub static LOSSY: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut b = 0;
    while b < 256 {
        let c = b as u8;
        t[b] = match c | 0x20 {
            b'a' | b'c' | b'g' | b't' => c,
            _ => b"ACTG"[((c >> 1) & 3) as usize],
        };
        b += 1;
    }
    t
};

// Returns whether `s` has ACGT bases only, in either case.
#[inline]
pub fn is_acgt(s: &[u8]) -> bool {
    s.iter().fold(true, |ok, &c| {
        let x = c | 0x20;
        ok & ((x == b'a') | (x == b'c') | (x == b'g') | (x == b't'))
    })
}

// Times of the index's parts: the file's scan, then the threads' work, summed over them.
#[derive(Default)]
pub struct Timings {
    pub scan: Duration,
    pub tm: Duration,
    pub tq: Duration,
    pub freeze: Duration,
}

// Indexes the patterns at `path` with `threads` threads: builds the minimizer index of their `m`-minimizers of
// `k`-mers, and hands each work item's sequences to its thread's k-mer sink, made by `make_sink`. Returns the
// minimizer index, with the time taken by each part.
pub fn ingest<S, F>(path: &str, k: usize, m: usize, threads: usize, make_sink: F) -> io::Result<(ShardedSet, Timings)>
where
    F: Fn() -> S + Sync,
    S: FnMut(&[&[u8]]),
{
    let mut timings = Timings::default();
    let start = Instant::now();
    let buf = Mapped::open(path)?;
    let recs = records(&buf, threads)?;

    // The runs, and the work items: consecutive runs of at least `ITEM_BYTES` in all.
    let mut runs = Vec::with_capacity(recs.len());
    for &(s, e) in &recs {
        let mut b = s;
        while e - b > ITEM_BYTES {
            runs.push(Run { start: b, end: b + ITEM_BYTES, rec_end: e, ext: true });
            b += ITEM_BYTES;
        }
        runs.push(Run { start: b, end: e, rec_end: e, ext: false });
    }
    let mut items = vec![0];
    let mut bytes = 0;
    for (i, r) in runs.iter().enumerate() {
        bytes += r.end - r.start;
        if bytes >= ITEM_BYTES {
            items.push(i + 1);
            bytes = 0;
        }
    }
    if *items.last().unwrap() < runs.len() {
        items.push(runs.len());
    }
    timings.scan = start.elapsed();

    let w = k - m + 1;
    let shards: Vec<Mutex<FxHashSet<u64>>> = (0..1 << SHARD_BITS).map(|_| Mutex::new(FxHashSet::default())).collect();
    let next = AtomicUsize::new(0);
    let parts: Vec<(Duration, Duration)> = thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut sink = make_sink();
                    let mut tm = ShardBuffers::new(&shards);
                    let mini_builder = minimizers(m, w);
                    let mut asc = Vec::new();
                    let mut seqs = Vec::new();
                    let mut packed = PackedSeqVec::default();
                    let mut mini_pos = Vec::new();
                    let (mut t_tm, mut t_tq) = (Duration::ZERO, Duration::ZERO);
                    loop {
                        let x = next.fetch_add(1, Ordering::Relaxed);
                        if x + 1 >= items.len() {
                            break;
                        }

                        // The item's sequences, without their line feeds.
                        let t_0 = Instant::now();
                        asc.clear();
                        seqs.clear();
                        for r in &runs[items[x]..items[x + 1]] {
                            let b = asc.len();
                            push_stripped(&mut asc, &buf[r.start..r.end]);
                            if r.ext {
                                // The next run's first `k - 1` bases.
                                let (mut need, mut p) = (k - 1, r.end);
                                while need > 0 && p < r.rec_end {
                                    if buf[p] != b'\n' {
                                        asc.push(buf[p]);
                                        need -= 1;
                                    }
                                    p += 1;
                                }
                            }
                            if asc.len() - b >= k {
                                seqs.push((b, asc.len()));
                            } else {
                                asc.truncate(b);
                            }
                        }

                        // Their minimizers, into the minimizer index.
                        for &(b, e) in &seqs {
                            packed.clear();
                            packed.push_ascii(&asc[b..e]);
                            mini_pos.clear();
                            mini_builder.run(packed.as_slice(), &mut mini_pos);
                            let p = packed.as_slice();
                            for &pos in &mini_pos {
                                tm.push(p.slice((pos as usize)..(pos as usize + m)).as_u64());
                            }
                        }

                        // Their k-mers, into the k-mer index, as 1.0.0 indexes them.
                        if !is_acgt(&asc) {
                            asc.iter_mut().for_each(|c| *c = LOSSY[*c as usize]);
                        }
                        let t_1 = Instant::now();
                        let slices: Vec<&[u8]> = seqs.iter().map(|&(b, e)| &asc[b..e]).collect();
                        sink(&slices);
                        let t_2 = Instant::now();
                        t_tm += t_1 - t_0;
                        t_tq += t_2 - t_1;
                    }
                    (t_tm, t_tq)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (a, b) in parts {
        timings.tm += a;
        timings.tq += b;
    }

    // Freeze the minimizer index, with room for half as many more minimizers again, as 1.0.0 leaves it.
    let start = Instant::now();
    let mut shards: Vec<FxHashSet<u64>> = shards.into_iter().map(|s| s.into_inner().unwrap()).collect();
    let per = shards.len().div_ceil(threads.max(1));
    thread::scope(|s| {
        for part in shards.chunks_mut(per) {
            s.spawn(move || {
                for set in part {
                    set.reserve(set.len() / 2);
                }
            });
        }
    });
    timings.freeze = start.elapsed();

    Ok((ShardedSet { shards }, timings))
}
