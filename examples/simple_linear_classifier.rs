use brains::ops::out;
use brains::dataset::{Dataset, LoadDatasetFromFile};
use ndarray::*;
use std::{error::Error, fs::File, path::Path};



#[derive(serde_derive::Deserialize)]
struct MNISTData {
    imgs: Vec<Vec<Vec<u8>>>,
    labels: Vec<u8>,
}

#[derive(serde_derive::Deserialize)]
struct MNIST {
    train: MNISTData,
    test: MNISTData,
}

impl LoadDatasetFromFile for MNIST { }

fn main() -> Result<(), Box<dyn Error>> {
    let B = 32;
    let F = 784;
    let C = 10;

    let model = brains::model("linear_classifier", |tf, input| {
        let w = tf.var::<f32>([F, C]);
        let b = tf.var::<f32>([F]);

        let x = input.required("x", [F]);

        let logits = x.matmul(w);
        let probabilities = logits.softmax();
        let y = probabilities.argmax();

        out! {
            default "y" => y,
            "logits" => logits,
            "probs" => probabilities,
        }
    })?;

    let loss = brains::function("categorical_nll_loss", |tf, input| {
        let logits = input.required("logits", [C]);
        let targ = input.required("targ", [C]);

        let mult_logits = targ * logits + (1 - targ) * (1 - logits);
        let loss = -mult_logits.sum();

        out! { loss }
    })?;

    let mnist = MNIST::load("./examples/7x7_mnist.data.json")?;

    let trainer = brains::TrainingLoop::new(model, loss)
        .with_optimizer(brains::sgd())
        .fit_to(mnist)?;

    Ok(())
}
