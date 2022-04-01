//activation functions are pure functions that define how each node/neuron is activated
use tensorflow::ops;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;

///a builder for passing in the various configurations needed for
/// different activation functions.

pub type Activation = Box<dyn Fn(Output, &mut Scope) -> Result<Operation, Status>>;
pub fn tanh(max_integer: u32) -> Activation {
    Box::new(move |output, scope| {
        Ok(ops::multiply(
            ops::tanh(output, scope)?,
            ops::constant(max_integer as f32, scope)?,
            scope,
        )?
        .into())
    })
}
pub fn sigmoid(max_integer: u32) -> Activation {
    Box::new(move |output, scope| {
        Ok(ops::multiply(
            ops::sigmoid(output, scope)?,
            ops::constant(max_integer as f32, scope)?,
            scope,
        )?
        .into())
    })
}

//fn relu
//etc..
