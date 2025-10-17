import click
import hydra
from pathlib import Path


@click.command(help="Collect expert demonstrations in simulation.")
@click.option("-t", "--task", type=str, required=True, help="Simulation task.")
@click.option("-r", "--runner", type=str, required=True, help="Policy runner.")
@click.option("-c", "--checkpoint", type=str, default=None, help="Checkpoints.")
@click.option("-s", "--seed", type=int, default=None, help="Initial seed.")
def main(task, runner, checkpoint, seed):
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).parent.joinpath("sim_demo_collector/config")),
        version_base=None
    ):
        cfg = hydra.compose(config_name="config", overrides=[f"task={task}"])
        runner = hydra.utils.instantiate(cfg.runner)
        runner.run(checkpoint, seed)


if __name__ == "__main__":
    main()