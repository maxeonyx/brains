use crate::ops::Outputs;

trait Block {
    fn get_name(&self) -> &str;
    fn get_inputs(&self) -> &Vec<BlockInput>;
}

fn function<F>(name: &str, f: F) -> Outputs {

}
