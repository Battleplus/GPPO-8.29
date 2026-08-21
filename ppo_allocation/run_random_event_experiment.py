"""Thin script entry point for :mod:`random_event.experiment`.

Examples (run from ``ppo_allocation``)::

    python run_random_event_experiment.py smoke
    python run_random_event_experiment.py protocol-bank --tier preliminary --split validation
    python run_random_event_experiment.py protocol-bank --tier preliminary --split test
    python run_random_event_experiment.py train --seeds 1,2,3 --timesteps 2000
    python run_random_event_experiment.py evaluate
"""

from random_event.experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
