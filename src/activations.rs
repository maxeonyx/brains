//activation functions are pure functions that define how each node/neuron is activated
use tensorflow::ops;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Status;

//TODO: can this be a macro? should it be?
///a builder for passing in the various configurations needed for
/// different activation functions.
pub fn tanh(max_integer: u32) -> Box<dyn Fn(Output, &mut Scope) -> Result<Operation, Status>> {
    Box::new(move |output, scope| {
        Ok(ops::multiply(
            ops::tanh(output, scope)?,
            ops::constant(max_integer as f32, scope)?,
            scope,
        )?
        .into())
    })
}
//TODO: sigmoid and other functions, most shouldnt need max_integer or other parameters
// fn sigmoid(output: Output, scope: &mut Scope) -> Result<Output, Status>{
//   Ok(ops::multiply(
//     ops::sigmoid(x, scope)?,
//     ops::constant(max_integer as f32, scope)?,
//     scope,
//   )?
//   .into())
// }
//fn relu
//etc..
