import os
from tcrpeg_toolkit.p_infer_calculation import TCRpegModel, configure_logging

# Load the configuration file
configfile: os.path.join(workflow.basedir, "..", "config", "p_infer_calculation_config.yaml")

# Use greedy scheduler due to complexity of the rules
scheduler = "greedy"

rule all:
    input:
        expand(config["output_dir"] + "/logs/{sample}_tcrpeg_p_infer.log",
               sample=glob_wildcards(config["input_dir"] + "/{sample}.csv").sample)

rule run_tcrpeg_infer:
    input:
        config["input_dir"] + "/{sample}.csv"
    output:
        # results = config["output_dir"] + "/{sample}_tcrpeg_results.csv",
        log = config["output_dir"] + "/logs/{sample}_tcrpeg_p_infer.log"
    run:
        configure_logging(output.log)
        model = TCRpegModel(
            input_file=str(input[0]),  # Ensure string path
            output_dir=str(config["output_dir"]),  # Ensure string path
            device=str(config.get("device", "cpu"))  # Ensure string device
        )
        model.run(
            seq_col=str(config.get("seq_col", "sequence")),
            count_col=str(config.get("count_col", "count")),
            id_col=str(config.get("id_col", "id")),
            test_size=float(config.get("test_size", 0.2)),
            word2vec_epochs=int(config.get("word2vec_epochs", 10)),
            word2vec_batch_size=int(config.get("word2vec_batch_size", 100)),
            word2vec_learning_rate=float(config.get("word2vec_learning_rate", 1e-4)),
            hidden_size=int(config.get("hidden_size", 128)),
            num_layers=int(config.get("num_layers", 5)),
            epochs=int(config.get("epochs", 20)),
            batch_size=int(config.get("batch_size", 100)),
            learning_rate=float(config.get("learning_rate", 1e-4))
        )
