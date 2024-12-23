import argparse
import os

import datasets
import pandas as pd

import actions
from agents import StructuredOutputDynamicActionSpaceAgent
from env import Env
from prompts import ACTION_DESCRIPTION_TEMPLATE, DYNASAUR_PROMPT
from scripts.llm_engines import AzureOpenAIEngine, StructuredOutputAzureOpenAIEngine
from scripts.reformulator import prepare_response
from scripts.run_agents import answer_questions


def get_dataset(args):
    dataset = datasets.load_dataset("gaia-benchmark/GAIA", args.split)[args.set]
    dataset = dataset.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "level", "Annotator Metadata": "annotations"})

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_path"] = os.path.join(args.data_dir, args.set, row["file_name"])
        return row

    dataset = dataset.map(preprocess_file_paths)
    dataset = dataset.sort("level")

    print(f"Loaded {args.set} set:")
    print(pd.Series(dataset["level"]).value_counts(sort=False))
    return dataset


def get_env(args):
    extra_env_vars = {
        "GENERATED_ACTION_DIR": args.generated_action_dir,
        "MODEL_NAME": args.model_name,
    }
    env = Env(extra_env_vars=extra_env_vars)

    # Load user-defined actions from disk to env
    action_code = open("actions.py", "r").read()
    output = env.step(action_code)
    if output.error:
        raise Exception(f"Error loading user-defined actions: {output.error}")

    return env


def get_agent(args, env):
    llm_engine = StructuredOutputAzureOpenAIEngine(model_name=args.model_name, response_format="thought_code")

    # Load initial actions
    required_actions = list(actions.get_required_actions(args.generated_action_dir).values())
    user_defined_actions = list(actions.get_user_defined_actions(args.model_name).values())
    initial_actions = required_actions + user_defined_actions

    disable_accum = (args.set == "test")
    agent = StructuredOutputDynamicActionSpaceAgent(
        llm_engine=llm_engine,
        tools=initial_actions,
        max_iterations=args.max_iterations,
        verbose=2,
        system_prompt=DYNASAUR_PROMPT,
        tool_description_template=ACTION_DESCRIPTION_TEMPLATE,
        generated_tool_dir=args.generated_action_dir,
        disable_accum=disable_accum,
        env=env,
    )

    return agent


def get_agent_call_function(args):
    llm_engine = AzureOpenAIEngine(args.model_name)

    def agent_call_function(agent, question: str, **kwargs) -> str:
        result = agent.run(question, **kwargs)

        agent_memory = agent.write_inner_memory_from_logs(summary_mode=True)
        try:
            final_result = prepare_response(question, agent_memory, llm_engine)
        except Exception as e:
            print(e)
            final_result = result

        # Try getting metrics if the Agent supports it
        metrics = {}
        if hasattr(agent, "metrics"):
            metrics = agent.metrics

        return {
            "output": str(final_result),
            "intermediate_steps": [
                {key: value for key, value in log.items() if key != "agent_memory"}
                for log in agent.logs
            ],
            "metrics": metrics,
        }

    return agent_call_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/gaia")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--generated_action_dir", type=str, default="generated_actions")
    parser.add_argument("--set", type=str, default="validation")
    parser.add_argument("--split", type=str, default="2023_level1")
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--max_iterations", type=int, default=20)
    args = parser.parse_args()

    agent_name = f"{args.model_name}-{args.split}"
    generated_action_dir = os.path.join(args.generated_action_dir, agent_name)
    args.agent_name = agent_name
    args.generated_action_dir = generated_action_dir
    print(f"EXP: {agent_name}")

    dataset = get_dataset(args)
    env = get_env(args)
    agent = get_agent(args, env)

    agent_call_function = get_agent_call_function(args)

    results = answer_questions(
        dataset,
        agent,
        agent_name,
        agent_call_function=agent_call_function,
        output_folder=f"{args.output_dir}/{args.set}",
    )
