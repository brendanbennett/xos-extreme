from concurrent.futures import process
from multiprocessing import Process, Queue
from sim import XOAgentModel, Network, MCTS, save_training_data

N_PROCESSES = 4
GAMES_PER_SAVE = 4
AGENT_REVISION = 0

def worker(save_queue: Queue, agent: XOAgentModel, games_per_save):
    monte = MCTS()
    while True:
        list_training_data = []
        for i in range(games_per_save):
            list_training_data.append(monte.self_play(agent, 200))
        save_queue.put(list_training_data)

def saver(save_queue: Queue):
    while True:
        list_training_data = save_queue.get(block=True)
        save_training_data(list_training_data, AGENT_REVISION)

if __name__ == "__main__":
    net = Network(2 + 1, 32, 4)
    agent = XOAgentModel(net)

    save_queue = Queue()
    processes: list[Process] = []

    for i in range(N_PROCESSES):
        processes.append(Process(target=worker, args=(save_queue, agent, GAMES_PER_SAVE), daemon=True))

    processes.append(Process(target=saver, args=(save_queue,), daemon=True))
    
    for i in range(len(processes)):
        processes[i].start()
    for i in range(len(processes)):
        processes[i].join()



    