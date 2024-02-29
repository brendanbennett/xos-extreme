from concurrent.futures import process
from multiprocessing import Process, Queue
from sim import XOAgentModel, Network, MCTS, save_training_data, get_agent
from torch import load

N_PROCESSES = 2
GAMES_PER_SAVE = 4

def worker(save_queue: Queue, games_per_save: int, agent: XOAgentModel):
    while True:
        list_training_data = []
        for i in range(games_per_save):
            monte = MCTS()
            list_training_data.append(monte.self_play(agent, 200))
        save_queue.put(list_training_data)

def saver(save_queue: Queue, revision: str):
    while True:
        list_training_data = save_queue.get(block=True)
        save_training_data(list_training_data, revision)

if __name__ == "__main__":
    save_queue = Queue()
    processes: list[Process] = []

    for i in range(N_PROCESSES):
        net = Network(hidden=128)
        agent, revision = get_agent(network=net)
        processes.append(Process(target=worker, args=(save_queue, GAMES_PER_SAVE, agent), daemon=True))

    processes.append(Process(target=saver, args=(save_queue, revision), daemon=True))
    
    for i in range(len(processes)):
        processes[i].start()
    for i in range(len(processes)):
        processes[i].join()



    