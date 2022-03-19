//TODO: allow this to fallback to TF-CPU for machines without GPU and possibly just do inference with a build script.
//TODO: floats stink norm net is an attempt to realizing rot net and uint8 operations with bounded functions while retaining all the flaws in DNNs.. (go back to rot net at some point)
//TODO: rayon all the things
//TODO: extract layers to layers.rs and upload to crates.io with example of how to implement an architecture

//TODO: type errors for anything other than u64 architecture due to casts. 
//      Also precision errors for summing f32s which should accumulate to f64.
//TODO: logging for effeciency, SED log::debug! to a logging crate since stdout logs are already formatted well
//NOTE: DONE WITH FEATURES. this is a basic layers based tensorflow backed ANN architecture framework. 
//      Anything else should be written outside this crate and integrated only if general enough.

//allow unstable features
// #![feature(int_log)]
mod layers;

use log;
use serde::{Serialize, Deserialize};
use serde_derive::{Serialize, Deserialize};
use serde_json::{Value};
use half::bf16;
use half::f16;
use std::cell::RefCell;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{Write, Read};
use std::io::ErrorKind;
use rand::seq::IteratorRandom;
use std::path::Path;
use std::result::Result;
use anyhow::{Context};
//TODO: clean this up with proper heirachy
use rand::Rng;
// include par_iter from rayon
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use std::os;
use std::rc::Rc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tensorflow::ops;
use tensorflow::train::AdadeltaOptimizer;
use tensorflow::train::GradientDescentOptimizer;
use tensorflow::train::MinimizeOptions;
use tensorflow::train::Optimizer;
use tensorflow::BFloat16;
use tensorflow::Code;
use tensorflow::DataType;
use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::OutputName;
use tensorflow::SavedModelBundle;
use tensorflow::SavedModelSaver;
use tensorflow::Scope;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Shape;
use tensorflow::SignatureDef;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::TensorInfo;
use tensorflow::Variable;
use tensorflow::REGRESS_INPUTS;
use tensorflow::REGRESS_METHOD_NAME;
use tensorflow::REGRESS_OUTPUTS;
use uuid::Uuid;
use layers::*;

//TODO: defaults such as learning rate
pub struct Brain<'a> {
    //TODO: refactor with builder pattern for RL, checkpointing and feedback network once tested and working.
    /// Tensorflow objects for user abstraction from Tensorflow
    scope: Scope,
    session: Session,
    session_options: SessionOptions,
    ///each layers output for the network
    net_layers: Vec<Output>,
    ///all the trainable parameters of the network
    net_vars: Vec<Variable>,
    ///Operations to interact with the graph (kept here for serialization)
    Input: Operation,
    Label: Operation,
    Output_op: Operation,
    Error: Operation,
    ///variables to be minimized
    minimize_vars: Vec<Variable>,
    ///regression operation
    minimize: Operation,
    ///class for serializing, saving and loading the model
    SavedModelSaver: RefCell<Option<SavedModelSaver>>,
    ///user specified name of this model
    name: &'a str,
    ///the highest score achieved (lowest error) as a sum of the error vector. 
    ///initialized to -1 since error can never be negative.
    lowest_error: f32,
    //TODO: checkpoint_window_size may be better as a slice again
    ///the error power for scaling the error gradient's pressure on the weights
    error_power: f32,
    ///the moving average window of labels trained on before the score is evaluated
    checkpoint_window: Vec<f32>,
    //TODO: should these be setters if not in constructor? lazy instantiation? yes
    //TODO: Option<checkpoint_window_size>
    //TODO: if we have sparse_state_buffer we should just calculate checkpoint_window
    //TODO: rename to indicate this is outputs not errors
    //TODO: should this also keep each error in the buffer or just use fractional sparse_reward? 
    //      Keep it simple for now, justify proportional error in sparse_state_buffer empirically.
    //stored inputs and outputs of the network for RL based experience replay.

    ///the current name of the checkpoint being searched
    checkpoint_name: Option<String>,
}
impl <'a>Brain <'a>{
    //TODO: type safety: some of the meta-learning values should be 64 bit for overflow since we sum 32s etc.
    pub fn new<Architecture>(
        name: &'a str,
        input_size: u64,
        output_size: u64,
        //Number of neurons in each layer
        layer_width: u64,
        //Number of layers in the network
        layer_height: u64,
        //TODO: move this into layer
        layer: Architecture,
        max_integer: u32,
        learning_rate: f32,
        error_power: f32,
    ) -> Result<Brain<'a>, Status> 
    where
        Architecture: 
            Fn(Output, u64, u64, &dyn Fn(Output, &mut Scope) -> Result<Operation, Status>,&mut Scope) -> 
            Result<(Vec<Variable>,Output, Operation), Status>{
        assert!(max_integer % 10 == 0 || max_integer == 1, "max_integer must be a multiple of 10 or 1 since it represents order of magnitude of the integer range");
        let mut scope = Scope::new_root_scope();

        // TODO: consider inlining this since were encapsulating
        let Input = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64, input_size])
            .build(&mut scope.with_op_name("input"))?;
        let Label = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64, output_size])
            .build(&mut scope.with_op_name("label"))?;

        //CONSTRUCT NETWORK
        //TODO: extract this to a builder pattern for constructor readability.
        let mut net_vars = vec![];
        let mut net_layers = vec![];

        //initial layer
        let (vars, layer, _) = norm_layer(
            Input.clone(),
            input_size,
            layer_width,
            &|x, scope| Ok(ops::tanh(x, scope)?.into()),
            &mut scope,
        )?;
        net_vars.extend(vars);
        net_layers.push(layer.clone());

        let mut prev_layer = layer;
        //hidden layers
        for i in 0..layer_height - 2 {
            let (vars, layer, _) = norm_layer(
                prev_layer.clone(),
                layer_width,
                layer_width,
                //NOTE: originally designed with tan but vanishing gradient can occur
                &|x, scope| Ok(ops::tanh(x, scope)?.into()),
                &mut scope,
            )?;
            prev_layer = layer.clone();

            net_vars.extend(vars);
            net_layers.push(layer.clone());
        }

        //the final output layer is tanh to express negative values and multiplied to stabilize the
        //half precision gradient as well as express whole integers outside of -1 and 1.
        let (vars, output, Output_op) = norm_layer(
            net_layers.last().unwrap().clone(),
            layer_width,
            output_size,
            &|x, scope| {
                Ok(ops::multiply(
                    ops::tanh(x, scope)?,
                    ops::constant(max_integer as f32, scope)?,
                    scope,
                )?
                .into())
            },
            &mut scope,
        )?;
        //NOTE: need net_vars and net_layers as return values
        net_vars.extend(vars);
        net_layers.push(output.clone());
        //END OF CONSTRUCTING NETWORK
        //TODO: END OF EXTRACTION

        let Output = output;

        let options = SessionOptions::new();
        let SavedModelSaver= RefCell::new(None);

        //TODO: pass this in? this should be modular so new implementations can be tried
         let mut optimizer = AdadeltaOptimizer::new();
         optimizer.set_learning_rate(ops::constant(learning_rate, &mut scope)?);
        // let mut optimizer =
        //     GradientDescentOptimizer::new(ops::constant(learning_rate, &mut scope)?);

        // DEFINE ERROR FUNCTION //
        //TODO: pass this in conditionally, give user output and label with
        //      a partial constructor then they supply error and construction is complete
        //      two structs? whats standard functionally for partial construction?
        //  ^this is important for reinforcement learning where label difference should be output gradient direction (multiplication)
        //TODO: use setters instead and have a default object behaviour.

        //default error is pythagorean distance
        let Error = ops::sqrt(
            ops::pow(
                ops::sub(Output.clone(), Label.clone(), &mut scope)?,
                ops::constant(2.0 as f32, &mut scope)?,
                &mut scope,
            )?,
            &mut scope,
        )?;
        let Error = ops::pow(
            Error.clone(),
            ops::constant(error_power, &mut scope).unwrap(),
            &mut scope.with_op_name("error"),
        )?;

        let mut lowest_error = f32::MAX;
        let mut checkpoint_window = vec![];

        let (minimize_vars, minimize) = optimizer
            .minimize(
                &mut scope,
                Error.clone().into(),
                MinimizeOptions::default().with_variables(&net_vars),
            )?
            .into();
        let session = Session::new(&options, &mut scope.graph())?;

        // set parameters to be optimization targets if they havent been set already
        let mut run_args = SessionRunArgs::new();
        for var in &net_vars {
            run_args.add_target(&var.initializer());
        }
        for var in &minimize_vars {
            run_args.add_target(&var.initializer());
        }
        session.run(&mut run_args)?;

        let mut init_norm_net = Brain {
            name,
            scope,
            session: session,
            session_options: options,
            net_layers,
            net_vars,
            Input,
            Label,
            Output_op,
            Error,
            error_power,
            minimize_vars,
            minimize,
            SavedModelSaver,
            lowest_error,
            checkpoint_window,
            checkpoint_name: None,
        };

        //initialize Brain save directory
        fs::create_dir_all(init_norm_net.name).unwrap();
        init_norm_net.save().unwrap();

        Ok(init_norm_net)
    }


    /// save variables that arent in the Tensorflow graph that are needed for checkpointing
    /// this stores edge information about which checkpoints resulted in which for informed 
    /// selection of checkpoints. Any future serialization should also be implemented here.
    fn serialize_network(&self, dir: String) -> Result<(), Box::<dyn Error>> {
        let mut name = "".to_string();
        let parent_search_name = self.checkpoint_name.clone();
        if parent_search_name.is_none(){
            //this is a root node of the checkpoint search graph
            name = "Root".to_string();
        }else{
            name = parent_search_name.unwrap();
        }
        // create a serialized_network object
        let serialized_network = SerializedNetwork{
            parent_search_name: name.clone(),
            //TODO: see if we can remove this option in Brain
            checkpoint_name: Some(dir.clone()),
            lowest_error: self.lowest_error,
        };

        //save the parent_search_name to dir which is where the model is saved, this is the edge to the checkpoint tree
        let file_name = format!("{}/checkpoint_data.json", dir);
        log::debug!("serializing non-tensorflow graph variables: {}", file_name);
        // append the file type .j
        // create the file
        let mut file = fs::File::create(file_name.clone())?;
        let serialized_network_string = serde_json::to_string(&serialized_network)?;
        // open the file and write name to it
        file.write_all(serialized_network_string.as_bytes())?;
        file.sync_all()?;

        Ok(())
    }

    /// stores the average error in self.lowest error if it is lower than the current lowest 
    /// average error and evaluation window is full, otherwise push error into checkpoint_window buffer.
    fn checkpoint_error(&mut self, error: Tensor<f32>, checkpoint_window_size: u64) ->  Result<(), Box::<dyn Error>>{
        // checkpoint window size must be > 0
        assert!(checkpoint_window_size > 0, "checkpoint_window_size must be > 0");
        // let mut result = false;

        // log::debug!("lowest error: {}", self.lowest_error);
        // print window length and checkpoint_window_size

        // log::debug!("window length: {}", self.checkpoint_window.len());
        // log::debug!("checkpoint_window_size: {}", checkpoint_window_size);
        //TODO: should we use checkpoint window size? there is a possibility of destroying the 
        //      best error in the window and still representing the now diverged solution.
        // the latent aspect of the gradient makes the solution still viable. it is "near" the 
        // solution by worst case radius(checkpoint_windw_size).
        // 
        // greedy checkpointing of epsilon > fitness is also great for sharp ridges in fitness landscape. this would be checkpoint window of 1.
        // checkpoint_window, as the name implies, can be modeled as a smoothing/clustering function.

        if self.checkpoint_window.len() >= checkpoint_window_size as usize {
            // log::debug!("evaluation window is full");
            // average checkpoint_window (dont need to divide since always comparing this value relatively) but we do so for more intuitive metrics relative to per column error
            let avg = self.checkpoint_window.iter().fold(0.0, |acc, x| acc + x)/ self.checkpoint_window.len() as f32;
            // print!("avg: {}", avg);
            // log::debug!(" vs lowest error: {}", self.lowest_error);

            // this will probably get LICM'd anyways:
            let error_sum = error.iter().fold(0.0, |acc, x| acc + x)/error.len() as f32;
            self.checkpoint_window.push(error_sum);

            // log::debug!("error sum: {}", error_sum);
            if self.lowest_error > avg {
                log::debug!("NEW LOWEST ERROR: {}", avg);
                self.lowest_error = avg;
                log::debug!("checkpointing..");
                self.save()?;
                log::debug!("new best score: {:?}", self.lowest_error);
            }
            // process the checkpoint_window buffer
            self.checkpoint_window.clear();
        }else{
            // only push the current error
            let error_sum = error.iter().fold(0.0, |acc, x| acc + x)/error.len() as f32;
            self.checkpoint_window.push(error_sum);

            log::debug!("error sum: {}", error_sum);
        }
        // return Ok with Box dyn error
        Ok(())
    }

    //TODO: include name score etc in serialization here (can just write to the directory created)
    //save the model out to disk in this directory as default
    pub fn save(&mut self) -> Result<(), Box<dyn Error>> {
        // save the model to disk in the current directory
        if self.SavedModelSaver.borrow().is_none() {
            log::debug!("initializing saved model saver..");
            let mut all_vars = self.net_vars.clone();
            all_vars.extend_from_slice(&self.minimize_vars);
            let mut builder = tensorflow::SavedModelBuilder::new();
            builder
                .add_collection("train", &all_vars)
                .add_tag("serve")
                .add_tag("train")
                .add_signature(REGRESS_METHOD_NAME, {
                    let mut def = SignatureDef::new(REGRESS_METHOD_NAME.to_string());
                    def.add_input_info(
                        REGRESS_INPUTS.to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.Input.name()?,
                                index: 0,
                            },
                        ),
                    );
                    def.add_input_info(
                        "label".to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.Label.name()?,
                                index: 0,
                            },
                        ),
                    );
                    //NOTE: we only need this for reporting learning position for learning rate
                    def.add_input_info(
                        "error".to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.Error.name()?,
                                index: 0,
                            },
                        ),
                    );
                    def.add_input_info(
                        "minimize".to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.minimize.name()?,
                                index: 0,
                            },
                        ),
                    );

                    def
                });
            let saved_model_saver = builder.inject(&mut self.scope)?;
            self.SavedModelSaver.replace(Some(saved_model_saver));
        }
        //TODO: annotate this with user input, fitness and checkpoint number
        // generate a uuid
        let uuid = Uuid::new_v4();
        // same as above but just dir, name and a uuid
        let cur_checkpoint_name = format!("{}/{}_{}", self.name, self.name, uuid);

        //update checkpoint_name
        log::debug!("saving model to {}", cur_checkpoint_name);
        self.SavedModelSaver.borrow_mut().as_mut().unwrap().save(
            &self.session,
            &self.scope.graph(),
            cur_checkpoint_name.clone(),
        )?;
        log::debug!("serializing non-graph variables to {}", cur_checkpoint_name);
        self.serialize_network(cur_checkpoint_name.clone())?;
        self.checkpoint_name = Some(cur_checkpoint_name);

        Ok(())
    }

    /// load the saved model in the directory dir and restore it in self, removing the 
    /// previous tensorflow graph and session.
    pub fn load(&mut self, dir: String) -> Result<(), Box<dyn Error>> {
        log::debug!("loading previously saved model..");
        let mut graph = Graph::new();
        //TODO: ensure we can access variables from graph or otherwise
        let bundle =
            SavedModelBundle::load(&self.session_options, &["serve", "train"], &mut graph, dir)?;
        let signature = bundle
            .meta_graph_def()
            .get_signature(REGRESS_METHOD_NAME)?
            .clone();

        self.session = bundle.session;
        self.Input =
            graph.operation_by_name_required(&signature.get_input("inputs")?.name().name)?;
        self.Label =
            graph.operation_by_name_required(&signature.get_input("label")?.name().name)?;
        self.Error =
            graph.operation_by_name_required(&signature.get_input("error")?.name().name)?;
        self.minimize =
            graph.operation_by_name_required(&signature.get_input("minimize")?.name().name)?;

        //TODO: clear sparse_state_buffer and checkpoint_window along with all other vars
        self.checkpoint_window.clear();
        self.lowest_error = std::f32::MAX;

        Ok(())
    }

    /// Train the network with the given inputs and labels (must be synchronized in index order)
    ///
    ///**PARAMETERS**:
    /// 
    /// * inputs: the inputs to the network as a flattened 1D vector
    /// 
    /// * labels: the labels for the inputs as a flattened 1D vector, must align index wise with inputs. (e.g. inputs[0] must be labeled as labels[0])
    /// 
    /// * error: the error operation to minimize with (e.g. pythagorean error sqrt(x^2 + y^2))
    /// 
    /// * learning_rate: the learning rate for the network (e.g. 0.001)
    pub fn train<T: tensorflow::TensorType>(
        // &mut self,
        &mut self,
        inputs: Vec<Vec<T>>,
        labels: Vec<Vec<T>>,
        // TODO: extract from class: pass these is instead of constructing
        // error: Operation,
        // learning_rate: f32,
    ) -> Result<Vec<Tensor<f32>>, Box<dyn Error>> {
        //TODO: randomize input and labels while k-folding
        assert_eq!(inputs.len(), labels.len());
        log::debug!("trainning..");
        let mut result = vec![];

        let mut input_tensor: Tensor<T> = Tensor::new(&[1u64, inputs[0].len() as u64]);
        let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels[0].len() as u64]);

        log::debug!("inputs.len(): {}", inputs.len());
        log::debug!("{}", inputs[0].len());
        log::debug!("{}", labels[0].len());

        let mut input_iter = inputs.into_iter();
        let mut label_iter = labels.into_iter();
        //TODO: shouldnt have to do this
        input_iter.next();
        label_iter.next();
        let mut i = 0;
        let mut avg_t = vec![];
        loop {
            // start a timer
            let start = Instant::now();

            i += 1;
            let input = input_iter.next();
            let label = label_iter.next();
            if input.is_none() || label.is_none() {
                break;
            }
            let input = input.unwrap();
            let label = label.unwrap();
            // now get input and label as slices
            let input = input.as_slice();
            let label = label.as_slice();
            // now assign the input and label to the tensor
            for i in 0..input.len() {
                input_tensor[i] = input[i].clone();
            }
            for i in 0..label.len() {
                label_tensor[i] = label[i].clone();
            }

            let mut run_args = SessionRunArgs::new();
            run_args.add_target(&self.minimize);

            let error_squared_fetch = run_args.request_fetch(&self.Error, 0);
            let output = run_args.request_fetch(&self.Output_op, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            run_args.add_feed(&self.Label, 0, &label_tensor);
            self.session.run(&mut run_args)?;

            let res: Tensor<f32> = run_args.fetch(error_squared_fetch)?;
            let output: Tensor<T> = run_args.fetch(output)?;

            // get how long has passed
            let elapsed = start.elapsed();
            avg_t.push(elapsed.as_secs_f32());

            // update the moving average for time
            let average = avg_t.iter().sum::<f32>() / avg_t.len() as f32;

            log::debug!(
                "training on {}\n input: {:?} label: {:?} error: {} output: {} seconds/epoch: {:?}",
                i, input, label, res, output,average 
            );
            //TODO: we dont need to call this here? this is exclusively for checkpointing
            // self.checkpoint_error(res.clone(), 0);

            result.push(res);
        }

        Ok(result)
    }

    //TODO: k-fold
    //TODO: search iterations should see different datasets/epochs of dataset (not actual epoch backprop) via k-folding
    //      also cross validate the k-fold
    //TODO: dataset structure with methods to yield k-fold and assertions for inputs and labels
    //TODO: checkpoint selection should be modular so a neural network or other intelligent statistical model can be apllied easily instead of.. this..

    ///trains on the given inputs and labels until search_iterations have been completed.
    ///if the network scores higher than delta_loss, the network is checkpointed 
    ///(serialized and saved to the given directory).
    pub fn train_checkpoint_search<T: tensorflow::TensorType>(&mut self, inputs: Vec<Vec<T>>, labels: Vec<Vec<T>>, checkpoint_window_size: u64) -> Result<(), Box<dyn Error>> {
        assert!(checkpoint_window_size < inputs.len() as u64, "evaluation window size must be less than input/output data");

        let mut input_tensor: Tensor<T> = Tensor::new(&[1u64, inputs[0].len() as u64]);
        let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels[0].len() as u64]);
        //TODO: this can create collisions in the solutions, need to k-mediods cluster the solutions (last hyperparameter I promise)

        //START OF TRAIN SUBROUTINE
        assert_eq!(inputs.len(), labels.len());
        log::debug!("trainning..");

        log::debug!("inputs.len(): {}", inputs.len());
        log::debug!("{}", inputs[0].len());
        log::debug!("{}", labels[0].len());

        let input_itl = inputs.clone();
        let label_itl = labels.clone();
        let mut input_iter = input_itl.iter();
        let mut label_iter = label_itl.iter();
        //TODO: shouldnt have to do this
        input_iter.next();
        label_iter.next();

        let mut i = 0;
        let mut avg_t = vec![];
        //TODO: save the root node of search properly (if it doesnt exist)
        loop{
            // start a timer
            let start = Instant::now();
            i += 1;
            let input = input_iter.next();
            let label = label_iter.next();

            if input.is_none() || label.is_none() {
                break;
            }
            let input = input.unwrap();
            let label = label.unwrap();
            // now get input and label as slices
            let input = input.as_slice();
            let label = label.as_slice();
            // now assign the input and label to the tensor
            for i in 0..input.len() {
                input_tensor[i] = input[i].clone();
            }
            for i in 0..label.len() {
                label_tensor[i] = label[i].clone();
            }

            let mut run_args = SessionRunArgs::new();
            run_args.add_target(&self.minimize);

            let error_squared_fetch = run_args.request_fetch(&self.Error, 0);
            let output = run_args.request_fetch(&self.Output_op, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            run_args.add_feed(&self.Label, 0, &label_tensor);
            self.session.run(&mut run_args)?;

            let res: Tensor<f32> = run_args.fetch(error_squared_fetch)?;
            let output: Tensor<T> = run_args.fetch(output)?;
            //END TRAIN SUBROUTINE

            // get how long has passed
            let elapsed = start.elapsed();
            avg_t.push(elapsed.as_secs_f32());

            // update the moving average for time
            let average = avg_t.iter().sum::<f32>() / avg_t.len() as f32;

            log::debug!(
                "training on {}\n input: {:?} label: {:?} error: {} output: {} time/epoch(ms): {:?}",
                i, input, label, res, output,average 
            );

            self.checkpoint_error(res.clone(), checkpoint_window_size)?;
        }

        Ok(())
    }

    ///randomly select a model from dir with selection_pressure for the models fitness
    pub fn load_checkpoint_search(&mut self, selection_pressure: f32) -> Result<(), Box<dyn Error>> {
        assert!(selection_pressure >= 0.0 && selection_pressure <= 1.0, "selection_pressure must be between 0 and 1");
        log::debug!("loading checkpointed models..");
        let files_iter = fs::read_dir(self.name)?.par_bridge();
        // .par_bridge();
        // create an iterator from files_iter

        let mut r = rand::thread_rng();

        //TODO: extract this to helper functions to abstract the tree search operations and buffer/cache checkpoint_data from disk
        //TODO: rayon this because tree search
        let targets: HashMap<String, SerializedNetwork> = files_iter.filter(|dir|
            dir.as_ref().unwrap().path().is_dir())
            .map(|dir| {
                let dir = dir.unwrap().path().to_str().unwrap().to_string();
                // get the file "checkpoint_data" from inside directory and read the first line
                let target = dir.clone()+r"\checkpoint_data.json";
                // same as above but with from_reader
                let deserialized_network: SerializedNetwork = serde_json::from_reader(fs::File::open(target.clone()).unwrap()).unwrap();
                // store the checkpoint data in the hashmap with directory as key
                (dir.to_string(), deserialized_network)
        })
        .collect();

        // find the lowest error in targets
        let mut lowest_error = f32::MAX;
        targets.iter()
        .filter(|(dir, serialized_network)| serialized_network.lowest_error > 0.0)
        .for_each(|(dir, serialized_network)| {
            if serialized_network.lowest_error < lowest_error{
                lowest_error = serialized_network.lowest_error;
            }
        });
        // find the highest error in targets
        let mut highest_error = 0.0;
        // we filter f32::MAX which is root node 
        //TODO: dont filter root node, handle otherwise (reseed the var parameter initialization)
        targets.iter()
        .filter(|(dir, serialized_network)| serialized_network.lowest_error < f32::MAX)
        .for_each(|(dir,serialized_network)| {
            if serialized_network.lowest_error > highest_error{
                highest_error = serialized_network.lowest_error.clone();
            }
        });

        let total_range = rand::thread_rng().gen_range(lowest_error..highest_error);
        //TODO: this doesnt allow sampling above the median
        // find the mean for the set {lowest_error, highest_error}
        let mean= (highest_error-lowest_error)/2.0 + lowest_error;
        // now move the mean to lowest error by selection_pressure
        let highest_error= mean - selection_pressure*(highest_error-lowest_error)/2.0;

        let selection_threshold = rand::thread_rng().gen_range(lowest_error..highest_error);

        let selection;
        if total_range > selection_threshold {
            selection = total_range - selection_threshold;
        }else {
            selection = selection_threshold;
        }

        // now select with selection_pressure a network based on fitness
        let (dir, serialized_network) = targets.into_iter().filter(|(dir, serialized_network)| {
            selection > serialized_network.lowest_error.clone()
        })
        // .inspect(|(dir, serialized_network)| {
        //     log::debug!("choosing from: {} {:?}", serialized_network.lowest_error, dir);
        // })
        .choose(&mut r).context("no checkpoint found")?;

        //TODO: search the tree for diversity either by looking at the expected future reward of a node (number of unique paths to unique frontier nodes) 
        //      or searching as horizontally as possible from the current checkpoint node with fitness pressure

        log::debug!("loading network with fitness: {:?}", serialized_network.lowest_error);
        self.load(dir)?;
        serialized_network.restore(self);
        // TODO: if Root node is selected, reinitialize all variables for random seed of network to ensure robust search

        Ok(())
    }
    /// forward pass and return the output of the network.
    pub fn infer<T: tensorflow::TensorType>(&self, inputs: Tensor<T>)-> Result<Tensor<T>, Box<dyn Error>>{
        let mut run_args = SessionRunArgs::new();
        let output = run_args.request_fetch(&self.Output_op, 0);
        run_args.add_feed(&self.Input, 0, &inputs);

        self.session.run(&mut run_args)?;

        let output: Tensor<T> = run_args.fetch(output)?;
        Ok(output)
    }

    /// Online reinforcement learning method.
    /// 
    /// Takes the given inputs and fitness function and backprops the network once, 
    /// returning the output output vector. 
    /// This should be called as an online (realtime) reinforcment learning technique 
    /// where labels can be formed given a fitness function.
    /// 
    ///PARAMETERS:
    /// returns the output from the network for online learning implementation.
    ///
    /// Fitness function takes a 1D tensor of length output and returns a label tensor of same shape.
    pub fn evaluate(
        &mut self, 
        inputs: Vec<f32>, 
        checkpoint_window_size: u64,
        //TODO: make this internal to class as builder so we dont clone the dyn ref alot ITL
        //TODO: refactor the names here to make this more intuitive
        fitness_function: Box<dyn Fn(&Tensor<f32>)->Tensor<f32>>) -> Result<Tensor<f32>, Box<dyn Error>>{

        // let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels.len() as u64]);
        // now assign the input and label to the tensor
        let mut input_tensor = Tensor::new(&[1u64, inputs.len() as u64]);
        for i in 0..inputs.len() {
            input_tensor[i] = inputs[i].clone();
        }

        // call infer to get the output
        let outputs = self.infer(input_tensor.clone())?;

        //create labels using fitness function
        let reward= fitness_function(&outputs);

        //backprop
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&self.minimize);

        //TODO: sparse reward (variable episodic automation) buffering goes here if implemented
        let error_squared_fetch = run_args.request_fetch(&self.Error, 0);
        // set output feed manually
        //TODO: runtime says we need to feed input again so create a placeholder tensor of 0
        run_args.add_feed(&self.Input, 0, &input_tensor);
        // run_args.add_feed(&self.Output_op, 0, &outputs);
        //TODO: need to pass in the expected future reward as the label
        run_args.add_feed(&self.Label, 0, &reward);
        self.session.run(&mut run_args)?;

        let cur_error: Tensor<f32> = run_args.fetch(error_squared_fetch)?;

        self.checkpoint_error(cur_error.clone(), checkpoint_window_size)?;

        //return the output
        Ok(outputs.clone())
    }
    //TODO: feedback network and rework name
    //TODO: unrolling feedback network
    // pub fn feedback_sequence_evaluate()
}

///The serialized representation of the model outside of tensorflow graph.
///Primarily used for checkpoint search meta-learner.
#[derive(Serialize, Deserialize,Debug)]
struct SerializedNetwork{
    /// The previous checkpoint that created this checkpoint for informed graph 
    /// selection checkpoint search.
    parent_search_name: String,
    checkpoint_name: Option<String>,
    lowest_error: f32,
} 
impl SerializedNetwork{
    /// Constructs a new serialized network from a checkpoint name and the lowest error
    fn new(parent_search_name: String, checkpoint_name: Option<String>, lowest_error: f32) -> Self{
        Self{
            parent_search_name,
            checkpoint_name,
            lowest_error,
        }
    }
    fn restore(self, norm_net: &mut Brain) {
        // let mut Brain = Brain.clone();
        norm_net.checkpoint_name = self.checkpoint_name;
        norm_net.lowest_error = self.lowest_error;
        // Brain
    }
}

//TODO: add assertions for more than just runtime error.
#[cfg(test)]
mod tests {
    #[test]
    fn test_net() {
        log::debug!("test_net");
        //call the main function
        use crate::*;

        //CONSTRUCTION//
        let mut norm_net = Brain::new("test_net",2, 1, 10, 10, norm_layer, 10, 1.0, 5 as f32).unwrap();

        //FITNESS FUNCTION//
        //TODO: auto gen labels from outputs and fitness function.

        //TRAIN//
        let mut rrng = rand::thread_rng();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..1000 {
            // instead of the above, generate either 0 or 1 and cast to f32
            let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
            let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];

            inputs.push(input);
            outputs.push(output);
        }

        norm_net.train(inputs, outputs).unwrap();
    }

    #[test]
    fn test_serialization() {
        //TODO: failing after checkpoint features
        log::debug!("test_serialization");
        //call the main function
        use crate::*;

        //CONSTRUCTION//
        let mut norm_net = Brain::new("test_serialization",2, 1, 20, 15, norm_layer, 10, 0.01, 5 as f32).unwrap();
        //TRAIN//
        let mut rrng = rand::thread_rng();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..10 {
            // instead of the above, generate either 0 or 1 and cast to f32
            let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
            let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];

            inputs.push(input);
            outputs.push(output);
        }

        norm_net.train(inputs.clone(), outputs.clone()).unwrap();

        // save the network
        norm_net
            .save()
            .unwrap();

        //load the network
        let mut path = "".to_string();
        //NOTE: dont ever call a string something else in your crates or someone I know will find you.
        let _ = std::path::PathBuf::new();
        for entry in fs::read_dir("test_serialization/").unwrap() {
            let entry = entry.unwrap();
            let is_dir = entry.path();
            if !is_dir.is_dir() {
                continue;
            } else {
                path = is_dir.clone().to_str().unwrap().to_string();
            }
            log::debug!("{:?}", path);
        }
        let path = path.to_string();
        log::debug!("{:?}", path);

        norm_net.load(path.to_string()).unwrap();

        norm_net.train(inputs, outputs).unwrap();
    }

    #[test]
    fn test_checkpoint(){
        log::debug!("test_checkpoint");
        use crate::*;
        //CONSTRUCTION//
        let mut norm_net = Brain::new("test_checkpoint",2, 1, 200, 96, norm_layer, 10, 10.0, 5 as f32).unwrap();
        //TRAIN//
        let mut rrng = rand::thread_rng();
        // create entries for inputs and outputs of xor
        //TODO: window size and training_iterations is hyperparameter for arch search. they should exist in shared struct or function parameter 
        //TODO: how can we train this in RL? need to store window and selection_pressure in class state
        //TODO: this needs to happen on initialization
        for _ in 0..15{
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..10 {
                // instead of the above, generate either 0 or 1 and cast to f32
                let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
                let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];

                inputs.push(input);
                outputs.push(output);
            }
            // TEST TRAIN
            norm_net.train_checkpoint_search(inputs.clone(), outputs.clone(),  2).unwrap();

            // TEST LOAD
            norm_net.load_checkpoint_search(0.001).unwrap();
        }
    }
    #[test]
    fn test_infer(){
        log::debug!("test_inference");
        use crate::*;
        //CONSTRUCTION//
        let mut norm_net = Brain::new("test_inference",2, 1, 200, 96, norm_layer, 10, 10.0, 5 as f32).unwrap();
        //TRAIN//
        let mut rrng = rand::thread_rng();
        // create entries for inputs and outputs of xor
        for _ in 0..10{
            let mut inputs:Tensor<f32> = Tensor::new(&[1u64, 2 as u64]);
            let res: Tensor<f32>= norm_net.infer(inputs).unwrap();
        }
    }
    #[test]
    fn test_evaluate(){
        log::debug!("test_evaluate");
        use crate::*;
        //CONSTRUCTION//
        let mut norm_net = Brain::new("test_evaluate",2, 1, 200, 96, norm_layer, 10, 1.0, 10 as f32).unwrap();
        //TRAIN//
        let mut rrng = rand::thread_rng();

        //TODO: this doesnt make much sense since we arent implementing state-action-reward tables 
        let fitness_function = Box::new(|outputs: &Tensor<f32>| -> Tensor<f32> {
            //TODO: which scalars in the output vector do we want to maximize? a good default fitness function 
            //      is a scalar constant to each entry in the vector or power term.
            //pass a function with a derivative that goes to zero or 1?
            let mut res_tensor = Tensor::new(outputs.dims());
            res_tensor[0] = outputs[0]/outputs[0];
            res_tensor
        });
        // create entries for inputs and outputs of xor
        for _ in 0..10{
            // same as above but push a vec with two random floats
            let inputs = vec![(rrng.gen::<f32>()), (rrng.gen::<f32>())];
            let res: Tensor<f32> =  norm_net.evaluate(inputs, 1, fitness_function.clone()).unwrap();
        }

        //SEARCH RECOVERED MODEL:
        norm_net.load_checkpoint_search(0.001).unwrap();
        for _ in 0..10{
            let inputs = vec![(rrng.gen::<f32>()), (rrng.gen::<f32>())];
            let res: Tensor<f32> =  norm_net.evaluate(inputs, 1, fitness_function.clone()).unwrap();
        }
    }
    //TODO: test layer modularity
    // #[test]
    // fn test_architectures(){}
}
