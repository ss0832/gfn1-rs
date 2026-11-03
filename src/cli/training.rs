// SPDX-License-Identifier: GPL-3.0-or-later
//! Training CLI. The calculation CLI remains an inference/property interface.
use crate::ml::{EnergyOptimizer, EnergyTrainingOptions, MlHamiltonian};
use crate::training::{fit_observables, EnergyLoss, EnergyWeighting, TrainingConfig};
use crate::{ElectronicOptions, Gfn1Error, Gfn1Parameters, PeriodicSystem, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
#[path = "train_support.rs"]
mod train_support;
use train_support::{dataset, evaluate, statistics, Dataset};

fn invalid(message: impl Into<String>) -> Gfn1Error {
    Gfn1Error::InvalidInput(message.into())
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Arguments(BTreeMap<String, String>);
impl Arguments {
    fn logger(&self) -> Result<Option<crate::CalculationLogger>> {
        self.0
            .get("--log-file")
            .map(|path| {
                crate::CalculationLogger::append(
                    path,
                    crate::LogLevel::parse(
                        self.0.get("--log-level").map_or("info", String::as_str),
                    )?,
                    true,
                )
            })
            .transpose()
    }
    fn parse(values: impl Iterator<Item = String>) -> Result<Self> {
        let mut values = values;
        let mut out = BTreeMap::new();
        while let Some(key) = values.next() {
            if !key.starts_with("--") {
                return Err(invalid(format!("unexpected training argument {key}")));
            }
            let value =
                if ["--train-correction", "--decay-bias", "--wtmad2"].contains(&key.as_str()) {
                    "true".into()
                } else {
                    values
                        .next()
                        .ok_or_else(|| invalid(format!("{key} requires a value")))?
                };
            if out.insert(key, value).is_some() {
                return Err(invalid("duplicate training option"));
            }
        }
        Ok(Self(out))
    }
    fn text(&self, key: &str) -> Result<&str> {
        self.0
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| invalid(format!("missing {key}")))
    }
    fn number<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        self.0.get(key).map_or(Ok(default), |s| {
            s.parse().map_err(|_| invalid(format!("invalid {key}")))
        })
    }
    fn flag(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    fn validate(&self, allowed: &[&str]) -> Result<()> {
        if let Some(key) = self.0.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(invalid(format!("unknown training option {key}")));
        }
        Ok(())
    }
    fn training(&self, preset_epochs: usize) -> Result<EnergyTrainingOptions> {
        let optimizer = match self.0.get("--optimizer").map_or("adamw", String::as_str) {
            "adamw" => EnergyOptimizer::AdamW {
                beta1: self.number("--beta1", 0.9)?,
                beta2: self.number("--beta2", 0.999)?,
                epsilon: self.number("--epsilon", 1e-8)?,
            },
            "sgd" => EnergyOptimizer::GradientDescent,
            _ => return Err(invalid("optimizer must be adamw or sgd")),
        };
        Ok(EnergyTrainingOptions {
            epochs: self.number("--epochs", preset_epochs)?,
            learning_rate: self.number("--learning-rate", 0.001)?,
            weight_decay: self.number("--weight-decay", 1e-8)?,
            optimizer,
            decay_bias: self.flag("--decay-bias"),
            train_correction: self.flag("--train-correction"),
        })
    }
    fn configuration(&self, data: &Dataset, preset_epochs: usize) -> Result<TrainingConfig> {
        let weighting = if self.flag("--wtmad2") {
            EnergyWeighting::Wtmad2
        } else {
            match self.0.get("--weighting").map_or("uniform", String::as_str) {
                "uniform" => EnergyWeighting::Uniform,
                "wtmad2" => EnergyWeighting::Wtmad2,
                _ => return Err(invalid("weighting must be uniform or wtmad2")),
            }
        };
        let loss = match self.0.get("--energy-loss").map(String::as_str).unwrap_or(
            if weighting == EnergyWeighting::Wtmad2 {
                "absolute"
            } else {
                "squared"
            },
        ) {
            "squared" => EnergyLoss::Squared,
            "absolute" => EnergyLoss::Absolute,
            "huber" => EnergyLoss::Huber {
                delta_hartree: self.number("--huber-delta", 0.001)?,
            },
            _ => return Err(invalid("energy loss must be squared, absolute or huber")),
        };
        let mut groups = data.groups.clone();
        if let Some(ratios) = self.0.get("--group-ratios") {
            for item in ratios.split(',') {
                let (name, value) = item.split_once(':').ok_or_else(|| {
                    invalid("group ratios must be NAME:FACTOR comma-separated pairs")
                })?;
                if name.trim().is_empty() {
                    return Err(invalid("empty subset name"));
                }
                let ratio = value.parse().map_err(|_| invalid("invalid subset ratio"))?;
                groups.entry(name.to_string()).or_default().ratio = ratio;
            }
        }
        Ok(TrainingConfig {
            optimizer: self.training(preset_epochs)?,
            energy_weight: self.number(
                "--energy-weight",
                if weighting == EnergyWeighting::Wtmad2 {
                    0.0
                } else {
                    1.0
                },
            )?,
            gradient_weight: self.number("--gradient-weight", 1.0)?,
            dipole_weight: self.number("--dipole-weight", 1.0)?,
            relative_energy_weight: self.number("--relative-energy-weight", 1.0)?,
            energy_loss: loss,
            energy_weighting: weighting,
            groups,
        })
    }
}
const FIT_OPTIONS: &[&str] = &[
    "--max-model-bytes",
    "--energy-weight",
    "--gradient-weight",
    "--dipole-weight",
    "--relative-energy-weight",
    "--weighting",
    "--wtmad2",
    "--energy-loss",
    "--huber-delta",
    "--group-ratios",
    "--log-file",
    "--log-level",
    "--model",
    "--data",
    "--output",
    "--report",
    "--param",
    "--epochs",
    "--learning-rate",
    "--weight-decay",
    "--optimizer",
    "--beta1",
    "--beta2",
    "--epsilon",
    "--decay-bias",
    "--train-correction",
    "--data-output",
];

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)?;
    Ok(())
}
fn electronic(temperature: f64) -> ElectronicOptions {
    ElectronicOptions {
        electronic_temperature: temperature,
        energy_tolerance: 1e-12,
        charge_tolerance: 1e-10,
        ..Default::default()
    }
}

fn fit(
    model: &mut MlHamiltonian,
    data: &Dataset,
    params: &Gfn1Parameters,
    args: &Arguments,
    preset_epochs: usize,
) -> Result<()> {
    let config = args.configuration(data, preset_epochs)?;
    let options = config.optimizer;
    let mut settings = electronic(data.temperature);
    settings.external_field.origin = data.origin;
    settings.logger = args.logger()?;
    let train = data.batch("train");
    let validation = data.batch("validation");
    let extrapolation = data.batch("extrapolation");
    let initial_sha = hash(model.to_model_string()?.as_bytes());
    let initial_file_sha = args
        .0
        .get("--model")
        .map(|path| fs::read(path).map(|b| hash(&b)))
        .transpose()?;
    model.training_provenance = format!(
        "gfn1-rs-train; dataset SHA-256 {}; reference {}; {}",
        data.sha256, data.reference_method, data.provenance
    );
    let output = PathBuf::from(args.text("--output")?);
    let report_path = args
        .0
        .get("--report")
        .map(PathBuf::from)
        .unwrap_or_else(|| output.with_extension("training.json"));
    if output == report_path {
        return Err(invalid("model and report paths must differ"));
    }
    let optimizer = match options.optimizer {
        EnergyOptimizer::GradientDescent => json!({"name":"sgd"}),
        EnergyOptimizer::AdamW {
            beta1,
            beta2,
            epsilon,
        } => json!({"name":"adamw","beta1":beta1,"beta2":beta2,"epsilon":epsilon}),
    };
    let mut report = json!({"schema":"gfn1-rs-training-report-v2","status":"training","package_version":env!("CARGO_PKG_VERSION"),
        "dataset_sha256":data.sha256,"initial_model_canonical_sha256":initial_sha,"initial_model_file_sha256":initial_file_sha,
        "parameter_sha256":hash(params.to_param_string().as_bytes()),"reference_method":data.reference_method,
        "dataset_provenance":data.provenance,"dataset_license_spdx":data.license,"independent_reference":data.independent,
        "electronic_temperature_K":data.temperature,"energy_definition":"mermin_free","dipole_origin_bohr":data.origin.to_array(),
        "gradient_method":"stationary energy and implicit CP-SCC mixed observable derivatives; no unrolled iterations",
        "optimizer":optimizer,"epochs":options.epochs,"learning_rate":options.learning_rate,"weight_decay":options.weight_decay,
        "decay_bias":options.decay_bias,"train_correction":options.train_correction,
        "energy_alignment":"one constant per predictor fitted only on training absolute-energy residual mean; none for relative cases"});
    let save = |report: &Value| {
        write(
            &report_path,
            &serde_json::to_vec_pretty(report).map_err(|e| invalid(e.to_string()))?,
        )
    };
    save(&report)?;
    let history = match fit_observables(
        model,
        params,
        &settings,
        &train.examples,
        &train.relative,
        &config,
    ) {
        Ok(h) => h,
        Err(error) => {
            report["status"] = json!("training_failed");
            save(&report)?;
            return Err(error);
        }
    };
    let text = model.to_model_string()?;
    let size = args.number("--max-model-bytes", usize::MAX)?;
    if text.len() > size {
        report["status"] = json!("model_size_limit_exceeded");
        save(&report)?;
        return Err(invalid("trained model exceeds --max-model-bytes"));
    }
    write(&output, text.as_bytes())?;
    report["status"] = json!("validating");
    report["model_sha256"] = json!(hash(text.as_bytes()));
    report["serialized_model_bytes"] = json!(text.len());
    report["weighting"] = history.weighting_metadata;
    report["training_loss"]=json!(history.epochs.iter().enumerate().map(|(i,l)|json!({"epoch":i+1,"total":l.total,"energy":l.energy,"gradient":l.gradient,"dipole":l.dipole,"relative_energy":l.relative_energy})).collect::<Vec<_>>());
    save(&report)?;
    let metrics = (|| -> Result<(Value, Value, Value, Option<[f64; 2]>)> {
        let base = evaluate(&train, params, &settings, None)?;
        let predicted = evaluate(&train, params, &settings, Some(model))?;
        let offsets = if base.absolute.is_empty() {
            None
        } else {
            Some([
                base.absolute.iter().sum::<f64>() / base.absolute.len() as f64,
                predicted.absolute.iter().sum::<f64>() / predicted.absolute.len() as f64,
            ])
        };
        let train_stats = statistics(&base, &predicted, &train, &config, offsets);
        let validation_stats = statistics(
            &evaluate(&validation, params, &settings, None)?,
            &evaluate(&validation, params, &settings, Some(model))?,
            &validation,
            &config,
            offsets,
        );
        let extrapolation_stats = statistics(
            &evaluate(&extrapolation, params, &settings, None)?,
            &evaluate(&extrapolation, params, &settings, Some(model))?,
            &extrapolation,
            &config,
            offsets,
        );
        Ok((train_stats, validation_stats, extrapolation_stats, offsets))
    })();
    let (train_stats, validation_stats, extrapolation_stats, offsets) = match metrics {
        Ok(v) => v,
        Err(error) => {
            report["status"] = json!("validation_failed");
            save(&report)?;
            return Err(error);
        }
    };
    let group_metrics = (|| -> Result<Value> {
        let mut result = serde_json::Map::new();
        for selected in ["train", "validation", "extrapolation"] {
            let mut groups = serde_json::Map::new();
            for (name, batch) in data.group_batches(selected) {
                groups.insert(
                    name,
                    statistics(
                        &evaluate(&batch, params, &settings, None)?,
                        &evaluate(&batch, params, &settings, Some(model))?,
                        &batch,
                        &config,
                        offsets,
                    ),
                );
            }
            result.insert(selected.into(), Value::Object(groups));
        }
        Ok(Value::Object(result))
    })();
    report["by_group"] = match group_metrics {
        Ok(v) => v,
        Err(error) => {
            report["status"] = json!("validation_failed");
            save(&report)?;
            return Err(error);
        }
    };
    report["group_metrics_definition"]=json!("Direct labels of this sample group plus relative cases touching its primary structures; shared partner labels excluded. A multigroup reaction may appear in multiple group diagnostics.");
    let improvement = |a: &str, b: &str| {
        if data.independent {
            validation_stats[a]
                .as_f64()
                .zip(validation_stats[b].as_f64())
                .map(|(baseline, trained)| trained < baseline)
        } else {
            None
        }
    };
    report["held_out_independent_shape_improved"] = json!(improvement(
        "baseline_shape_rmse_hartree",
        "model_shape_rmse_hartree"
    ));
    report["held_out_independent_relative_energy_improved"] = json!(improvement(
        "baseline_relative_rmse_hartree",
        "model_relative_rmse_hartree"
    ));
    report["held_out_independent_gradient_improved"] = json!(improvement(
        "baseline_gradient_rmse_hartree_per_bohr",
        "model_gradient_rmse_hartree_per_bohr"
    ));
    report["held_out_independent_dipole_improved"] = json!(improvement(
        "baseline_dipole_rmse_e_bohr",
        "model_dipole_rmse_e_bohr"
    ));
    report["train"] = train_stats;
    report["validation"] = validation_stats;
    report["extrapolation"] = extrapolation_stats;
    report["energy_alignment_offsets_hartree"] = json!(offsets);
    report["status"] = json!("complete");
    save(&report)?;
    if let Some(logger) = &settings.logger {
        logger.event(crate::LogLevel::Info,"training_validation",json!({"train":report["train"],"validation":report["validation"],"held_out_independent_shape_improved":report["held_out_independent_shape_improved"]}))?;
    }
    println!("model_sha256 {}", hash(text.as_bytes()));
    println!("serialized_model_bytes {}", text.len());
    println!("train {}", report["train"]);
    println!("validation {}", report["validation"]);
    println!(
        "held_out_independent_shape_improved {}",
        report["held_out_independent_shape_improved"]
    );
    Ok(())
}

pub(super) fn run(arguments: Vec<String>) -> Result<()> {
    let mut args = arguments.into_iter();
    let command = args.next().unwrap_or_else(|| "--help".into());
    if ["--version", "-V"].contains(&command.as_str()) {
        println!("gfn1-rs-train {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if ["--help", "-h", "help"].contains(&command.as_str()) {
        println!(
            "gfn1-rs-train new --species H,O --centers 1.2,2.5 --cutoff 6 --output initial.model"
        );
        println!("gfn1-rs-train fit --model initial.model --data dataset.json --output trained.model [--report report.json]");
        println!("gfn1-rs-train fit-properties --model initial.model --data properties.json --parameters Z:HEAD:COORD,... --output trained.model [--report report.json]");
        println!("Property objectives: exact unit-tagged analytic observables; --gradient-method auto|parameter-central, --parameter-step X, --regularization-to-initial X. HEAD=s|p|d|correction; COORD=0 bias, 1.. radial weights.");
        println!("gfn1-rs-train demo-h2 --output demo.model [--data-output synthetic.json]");
        println!("Training: --optimizer adamw|sgd --epochs N --learning-rate X --weight-decay X --beta1 X --beta2 X --epsilon X --decay-bias --train-correction");
        println!("Targets: --energy-weight X --gradient-weight X --dipole-weight X --relative-energy-weight X");
        println!("Subsets: --weighting uniform|wtmad2 --group-ratios NAME:FACTOR,NAME:FACTOR --energy-loss squared|absolute|huber --huber-delta X");
        println!("Model size: --max-model-bytes N. WTMAD-2 requires explicit relative/reaction energy cases.");
        println!("Logging: --log-file PATH --log-level error|info|debug|trace");
        println!("Resources: --threads N --memory-mb N (checked workspace budget in MiB)");
        println!("CP: --cp-tolerance X --cp-relative-tolerance X --cp-max-iter N");
        println!("demo-h2 is a synthetic Hamiltonian architecture check, not independent chemical-accuracy evidence.");
        return Ok(());
    }
    let args = Arguments::parse(args)?;
    if command == "new" {
        args.validate(&[
            "--species",
            "--centers",
            "--cutoff",
            "--width",
            "--limit",
            "--output",
        ])?;
        let species = args
            .text("--species")?
            .split(',')
            .map(|s| {
                s.parse::<u8>()
                    .ok()
                    .or_else(|| crate::symbol_to_z(s))
                    .ok_or_else(|| invalid("unknown model element"))
            })
            .collect::<Result<Vec<_>>>()?;
        let centers = args
            .text("--centers")?
            .split(',')
            .map(|x| x.parse().map_err(|_| invalid("invalid radial center")))
            .collect::<Result<Vec<f64>>>()?;
        let mut model = MlHamiltonian::zero(species, centers, args.number("--cutoff", 6.0)?);
        model.radial_width_bohr = args.number("--width", 1.0)?;
        model.maximum_level_shift_hartree = args.number("--limit", 0.05)?;
        let text = model.to_model_string()?;
        write(Path::new(args.text("--output")?), text.as_bytes())?;
        println!("model_sha256 {}", hash(text.as_bytes()));
        return Ok(());
    }
    if command == "fit-properties" {
        args.validate(&[
            "--model",
            "--data",
            "--output",
            "--report",
            "--param",
            "--parameters",
            "--epochs",
            "--learning-rate",
            "--parameter-step",
            "--maximum-parameter-step",
            "--max-backtracks",
            "--gradient-tolerance",
            "--loss-tolerance",
            "--regularization-to-initial",
            "--gradient-method",
            "--max-model-bytes",
            "--log-file",
            "--log-level",
        ])?;
        let params = Gfn1Parameters::resolve(args.0.get("--param").map(String::as_str))?;
        let (mut model, identity) = MlHamiltonian::resolve_with_identity(args.text("--model")?)?;
        let bytes = fs::read(args.text("--data")?)?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| invalid(format!("invalid property dataset JSON: {e}")))?;
        let mut data = crate::property_training::PropertyDataset::from_json(&value)?;
        data.source_sha256 = Some(hash(&bytes));
        let logger = args.logger()?;
        for sample in &mut data.samples {
            sample.electronic.logger = logger.clone();
        }
        let settings = crate::property_training::PropertyTrainingOptions::from_json(&json!({
            "parameters":args.text("--parameters")?.split(',').map(str::to_string).collect::<Vec<_>>(),
            "max_epochs":args.number("--epochs",50usize)?,"learning_rate":args.number("--learning-rate",0.1)?,
            "parameter_difference_step":args.number("--parameter-step",1e-5)?,
            "maximum_parameter_step":args.number("--maximum-parameter-step",0.05)?,
            "maximum_backtracks":args.number("--max-backtracks",16usize)?,
            "gradient_tolerance":args.number("--gradient-tolerance",1e-8)?,"loss_tolerance":args.number("--loss-tolerance",1e-14)?,
            "regularization_to_initial":args.number("--regularization-to-initial",0.)?,
            "gradient_method":args.0.get("--gradient-method").map_or("auto",String::as_str)
        }))?;
        let output = PathBuf::from(args.text("--output")?);
        let report = args
            .0
            .get("--report")
            .map(PathBuf::from)
            .unwrap_or_else(|| output.with_extension("property-training.json"));
        if output == report {
            return Err(invalid("model and report paths must differ"));
        }
        eprintln!(
            "property training warm-start model: {}",
            identity.description()
        );
        let mut result =
            crate::property_training::fit_properties(&mut model, &params, &data, &settings)?
                .as_json();
        let text = model.to_model_string()?;
        if text.len() > args.number("--max-model-bytes", usize::MAX)? {
            return Err(invalid("trained model exceeds --max-model-bytes"));
        }
        result["initial_model_identity"] = identity.as_json();
        result["saved_model_path"] = json!(output);
        result["saved_model_file_sha256"] = json!(hash(text.as_bytes()));
        write(&output, text.as_bytes())?;
        write(
            &report,
            &serde_json::to_vec_pretty(&result).map_err(|e| invalid(e.to_string()))?,
        )?;
        println!("model_sha256 {}", hash(text.as_bytes()));
        println!("termination {}", result["termination"]);
        println!("train {}", result["final_train"]);
        println!("validation {}", result["final_validation"]);
        if let Some(logger) = logger {
            logger.event(crate::LogLevel::Info, "property_training", result)?;
        }
        return Ok(());
    }
    args.validate(FIT_OPTIONS)?;
    let params = Gfn1Parameters::resolve(args.0.get("--param").map(String::as_str))?;
    match command.as_str() {
        "fit" => {
            let bytes = fs::read(args.text("--data")?)?;
            let value = serde_json::from_slice(&bytes)
                .map_err(|e| invalid(format!("invalid dataset JSON: {e}")))?;
            let data = dataset(&value, hash(&bytes))?;
            let mut model =
                MlHamiltonian::from_model_str(&fs::read_to_string(args.text("--model")?)?)?;
            fit(&mut model, &data, &params, &args, 100)
        }
        "demo-h2" => {
            let mut teacher = MlHamiltonian::zero(vec![1], vec![1.2, 2.5], 6.0);
            let head = &mut teacher.heads.get_mut(&1).unwrap().levels[0];
            head.bias = 0.0012;
            head.weights = vec![0.002, -0.001];
            let mut options = electronic(0.0);
            options.ml_model = Some(Arc::new(teacher));
            let mut samples = Vec::new();
            for (split, distances) in [
                ("train", vec![0.60, 0.70, 0.80, 0.90, 1.10, 1.25, 1.40]),
                ("validation", vec![0.65, 1.0, 1.32]),
                ("extrapolation", vec![1.8, 2.2]),
            ] {
                for distance in distances {
                    let xyz = format!("2\nsynthetic H2\nH 0 0 0\nH {distance} 0 0\n");
                    let system = PeriodicSystem::from_xyz_str(&xyz, 0.0, false)?;
                    let energy =
                        crate::run_electronic(&system, &params, options.clone())?.total_free;
                    samples
                        .push(json!({"xyz":xyz,"charge":0,"energy_hartree":energy,"split":split}));
                }
            }
            let value = json!({"schema":"gfn1-rs-energy-training-v1","energy_units":"hartree","energy_definition":"mermin_free","coordinate_units":"angstrom",
                "electronic_temperature_K":0.0,"provenance":"Independent synthetic GFN1 teacher; H s-head bias 0.0012 Eh, radial weights [0.002,-0.001] Eh, centers [1.2,2.5] bohr; generated by gfn1-rs-train demo-h2",
                "license_spdx":"GPL-3.0-or-later","reference_method":"synthetic GFN1 Hamiltonian","independent_reference":false,"samples":samples});
            let bytes = serde_json::to_vec_pretty(&value).map_err(|e| invalid(e.to_string()))?;
            if let Some(path) = args.0.get("--data-output") {
                write(Path::new(path), &bytes)?;
            }
            let data = dataset(&value, hash(&bytes))?;
            let mut model = MlHamiltonian::zero(vec![1], vec![1.2, 2.5], 6.0);
            fit(&mut model, &data, &params, &args, 300)
        }
        _ => Err(invalid(
            "unknown training subcommand; use gfn1-rs-train --help",
        )),
    }
}
