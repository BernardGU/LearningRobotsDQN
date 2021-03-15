import os
import sys
import getopt

import logging

from learning_agent import LearningAgent
from replay_memory import ReplayMemory
from trainer import Trainer
from human_controller import HumanController
from utils import Hyperparameters

logging.basicConfig(level=logging.INFO)
H = Hyperparameters()
n_threads = (12, 12) # (loader threads, saver threads)

def play_game(game, session_name, replay, collect_samples):
    assert n_threads[1] > 0
    
    trainer = Trainer(
        game=game,
        session_name=session_name,
        root=f'data/{game}/sessions',
        H=H,
    )
    controller = HumanController(trainer.env.action_space.n)
    try:
        replay_memory = None
        if collect_samples:
            replay_memory = ReplayMemory(
                capacity=20000,
                state_shape=(H.AGENT_HISTORY_LENGTH+1,84,84),
                batch_size=H.MINIBATCH_SIZE,
                preload_batches=0,
                n_loaders=0,
                n_savers=n_threads[0] + n_threads[1],
                name=replay,
                root=f'./data/{game}/transitions'
            )
        trainer.play(
            replay_memory=replay_memory,
            human_controller=controller,
            collect_samples=collect_samples,
            manual_control=True,
        )
    finally:
        trainer.save_trainer()
        if replay_memory is not None:
            replay_memory.stop()


def train_with_demonstrations(game, session_name, replay, checkpoint):
    assert n_threads[0] > 0
    assert n_threads[1] > 0
    
    trainer = Trainer(
        game=game,
        session_name=session_name,
        root=f'data/{game}/sessions',
        H=H,
    )

    if checkpoint is not None:
        trainer.learning_agent.load_agent(os.path.join(checkpoint, trainer.learning_agent.name))

    try:
        replay_memory = ReplayMemory(
            capacity=H.REPLAY_MEMORY_SIZE,
            state_shape=(H.AGENT_HISTORY_LENGTH+1,84,84),
            batch_size=H.MINIBATCH_SIZE,
            preload_batches=2,
            n_loaders=n_threads[0],
            n_savers=n_threads[1],
            name=replay,
            root=f'./data/{game}/transitions'
        )
        trainer.train_with_demonstrations(replay_memory)
    finally:
        trainer.save_trainer()
        replay_memory.stop()

def train_from_experience(game, session_name, replay, checkpoint):
    assert n_threads[0] > 0
    assert n_threads[1] > 0

    trainer = Trainer(
        game=game,
        session_name=session_name,
        root=f'data/{game}/sessions',
        H=H,
    )

    if checkpoint is not None:
        trainer.learning_agent.load_agent(os.path.join(checkpoint, trainer.learning_agent.name))

    try:
        replay_memory = ReplayMemory(
            capacity=H.REPLAY_MEMORY_SIZE,
            state_shape=(H.AGENT_HISTORY_LENGTH+1,84,84),
            batch_size=H.MINIBATCH_SIZE,
            preload_batches=2,
            n_loaders=n_threads[0],
            n_savers=n_threads[1],
            name=replay,
            root=f'./data/{game}/transitions'
        )
        trainer.train_with_experience(replay_memory)
    finally:
        trainer.save_trainer()
        replay_memory.stop()

def evaluate_agent(game, session_name, checkpoint):
    trainer = Trainer(
        game=game,
        session_name=session_name,
        root=f'data/{game}/sessions',
        H=H,
    )

    if checkpoint is not None:
        trainer.learning_agent.load_agent(os.path.join(checkpoint, trainer.learning_agent.name))

    # Evaluate 1000 times and save csv
    for i in range(1000):
        episodes, steps, reward = trainer.evaluate(num_episodes=30, eps=0.05)
        stats = {'episodes': episodes, 'steps': steps, 'reward': reward}
        print(f'Eval #{i}: {stats}')
        trainer.save_eval_results(stats)

def print_usage():
    print( "USAGE:\n"
          f"python {sys.argv[0]} -g <GAME> -s <SESSION_NAME> -a <ACTION> [-r <REPLAY_FOLDER> -c <CHECKPOINT_FOLDER>]\n"
          f"\nUSAGE EXAMPLES:\n"
           "- To play a game using session 'gravitar_play' and collect samples in the memory replay folder 'human_samples'\n"
          f"   python {sys.argv[0]} -g Gravitar -s gravitar_play -a collect -r human_samples\n"
           "- To train using session 'gravitar_test_demo' and the memory replay folder 'human_samples'\n"
          f"   python {sys.argv[0]} -g Gravitar -s gravitar_test_demo -a train_demo -r human_samples\n"
           "- To train using session 'gravitar_test' with default memory replay folder\n"
          f"   python {sys.argv[0]} -g Gravitar -s gravitar_test -a train_xp\n"
    )

if __name__ == '__main__':
    game = None
    session = None
    action = None
    replay = None
    checkpoint = None

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hg:s:a:r:c:",["game=","session=","action=","replay=","checkpoint="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h'):
            print_usage()
            sys.exit(0)
        elif opt in ('-g', 'game='):
            game = arg
        elif opt in ('-s', 'session='):
            session = arg
        elif opt in ('-a', 'action='):
            action = arg
        elif opt in ('-r', 'replay='):
            replay = arg
        elif opt in ('-c', 'checkpoint='):
            checkpoint = arg

    try:
        assert game is not None
        assert session is not None
        assert action is not None
        
        if action == 'play':
            play_game(game, session, None, collect_samples=False)
        elif action == 'collect':
            assert replay is not None
            play_game(game, session, replay, collect_samples=True)
        elif action == 'train_demo':
            assert replay is not None
            train_with_demonstrations(game, session, replay, checkpoint)
        elif action == 'train_xp':
            assert replay is not None
            train_from_experience(game, session, replay, checkpoint)
        elif action == 'eval':
            evaluate_agent(game, session, checkpoint)
        else:
            print_usage()
            sys.exit(0)
    except AssertionError:
        print_usage()
        sys.exit(2)

