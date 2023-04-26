mod outputs;
pub use outputs::*;
mod blocks;
pub use blocks::*;

use tensorflow::Operation;

trait Dims<const N_DIMS: usize> {
    fn n_dims(&self) -> usize;

    /// Batch dims are for vectorization, and are usually elided
    fn n_batch_dims(&self) -> usize;
    /// Sequence dims are implementation-specific, but include width/height in
    /// images, time in NLP and timeseries data, or further dimensions in tabular
    /// data.
    fn n_seq_dims(&self) -> usize;
    /// There is typically only one channel dimension, however it sometimes makes sense
    /// to have multiple, for example: heads in multi-head attention. (Especially when
    /// using "talking heads")
    fn n_channel_dims(&self) -> usize;

    fn shape(&self) -> [usize; N_DIMS];
}

trait TFOpWrapper<const N_DIMS: usize, T: Dims<N_DIMS>> {

}

// Wrapper for tensorflow::Operation
struct Op {
    operation: Operation,
}
