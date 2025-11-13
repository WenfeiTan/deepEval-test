# run_eval.py
from agent.weather_agent import weather_agent
from data.dataset import dataset

def main():
    for g in dataset.evals_iterator():
        weather_agent(g.input)

    print("Eval finished. Check DeepEval report.")

if __name__ == "__main__":
    main()
