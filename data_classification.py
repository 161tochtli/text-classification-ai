import asyncio
import openai
import pandas as pd
from tqdm.asyncio import tqdm
import logging
import re
import ast
from dotenv import load_dotenv
import os
import argparse
import yaml


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv(".secrets/.env")
load_dotenv(".config")

# Get the working directory from the .env file
work_dir = os.getenv('WORK_DIR')

# Initialize OpenAI API client
client = openai.AsyncOpenAI()

# Global counter and lock
processed_count = 0
lock = asyncio.Lock()

semaphore = asyncio.Semaphore(20)

async def process_row_async(row, system_template, user_template, param_cols, fixed_params, categories, model):
    """
    Process a single row using OpenAI API asynchronously.
    """
    global processed_count

    cols_items = row[param_cols].to_dict()

    # Build the system and user prompts
    system_prompt = parametrize_prompt(system_template, **cols_items, **fixed_params)
    user_prompt = parametrize_prompt(user_template, **cols_items, **fixed_params)
    system_messages = [{"role": "system", "content": system_prompt}]
    user_messages = [{"role": "user", "content": user_prompt}]

    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=system_messages + user_messages,
            )

            content = response.choices[0].message.content

            if args.one_hot_encoding:
                match = re.search(r"\[(.*?)\]", content)
                if not match:
                    raise ValueError("Invalid response format. Could not find one-hot encoding.")

                one_hot = [int(x) for x in match.group(1).split(",")]
                if len(one_hot) != len(categories):
                    raise ValueError(
                        f"Mismatch between one-hot output length and categories. Expected {len(categories)}, got {len(one_hot)}.")
                result = one_hot
            else:
                result = content

            # Update global counter with a lock
            async with lock:
                global processed_count
                processed_count += 1
                logging.info(f"Processed {processed_count} comments.")

            return result
    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return [0] * len(categories) if args.one_hot_encoding else None # Return zeros in case of an error


def parametrize_prompt(template, **kwargs):
    return template.format(**kwargs)


async def process_dataframe_with_openai_async(
        dataframe,
        system_template,
        user_template,
        param_cols,
        fixed_params,
        categories,
        model,
        batch_size,
        one_hot_encoding
):
    """
    Process a dataframe by classifying text into one-hot encoded categories using OpenAI API.
    """
    # Load templates
    with open(system_template, "r") as file:
        system_template = file.read()
        system_template = system_template + \
        """Por ejemplo, si la entrada pertenece a "Category1", responde: Category1
        Responde únicamente con la categoría, sin explicaciones adicionales ni texto extra."""
        if one_hot_encoding:
            system_template = system_template + \
            """Deben aparecer en el mismo orden que {categories} y en formato one-hot: e.g. [0,1,...,0]"""

    with open(user_template, "r") as file:
        user_template = file.read()

    # Create asynchronous tasks for each row
    tasks = [
        process_row_async(row, system_template, user_template, param_cols, fixed_params, categories, model)
        for _, row in dataframe.iterrows()
    ]

    # Process tasks in batches
    logging.info(f"Starting row processing with batch size: {batch_size}")
    results = []

    # Manual tqdm for async generator
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="Processing rows", ncols=100) as pbar:
        async for batch_result in process_in_batches(tasks, batch_size):
            results.extend(batch_result)
            pbar.update(1)  # Update progress bar after processing each batch

    if one_hot_encoding:
        # Add one-hot encoded columns to the dataframe
        results = [ast.literal_eval(i) for i in results]
        one_hot_df = pd.DataFrame(results, columns=categories)
        dataframe = pd.concat([dataframe.reset_index(drop=True), one_hot_df], axis=1)
    else:
        # Add classification column to the dataframe
        dataframe["classification"] = results
    logging.info("Row processing complete.")
    return dataframe



async def process_in_batches(tasks, batch_size=100):
    """
    Process tasks in batches to limit memory usage and API throttling.
    """
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        yield await asyncio.gather(*batch)


def load_task_configuration(task_name):
    with open(work_dir+f"llm_tasks/{task_name}/task_configuration.yaml", "r") as f:
        return yaml.safe_load(f)


def main(task_config):
    """
    Main function to load data, process it, and save the results.
    """

    # Load the data
    logging.info("Loading data...")
    df = pd.read_csv(work_dir+task_config.get("input_file"))

    # Reprocess if specified
    if task_config.get("reprocess"):
        logging.info("Reprocessing data to filter empty classifications...")
        classes = task_config.get("fixed_params").get(task_config.get("reprocess"))
        empty_classes = df[classes].sum(axis=1) == 0
        df = df[empty_classes]
        df = df[classes]

    # Extract arguments
    fixed_params = task_config.get("fixed_params")
    param_cols = task_config.get("param_cols")

    # Process the dataframe asynchronously
    logging.info("Starting processing of the dataframe...")
    processed_df = asyncio.run(
        process_dataframe_with_openai_async(
            dataframe=df,
            system_template= work_dir + f"llm_tasks/{task_config.get('task')}/system_prompt.txt",
            user_template= work_dir + f"llm_tasks/{task_config.get('task')}/user_prompt.txt",
            param_cols=param_cols,
            fixed_params=fixed_params,
            categories=task_config.get("fixed_params").get("categories",[]),
            model=task_config.get("model"),
            batch_size=task_config.get("batch_size"),
            one_hot_encoding=task_config.get("one_hot_encoding")
        )
    )

    # Add an "updated_at" column to the DataFrame
    processed_df["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save the processed DataFrame
    logging.info("Saving the processed data...")
    processed_df.to_csv(work_dir+task_config.get("output_file"), index=False)
    logging.info(f"Data saved to {task_config.get('output_file')}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load task configured in task file.")

    # Add task file argument
    parser.add_argument("--task", help="Name of the task to perform.")
    args = parser.parse_args()

    # Load task file
    task_config = load_task_configuration(args.task)

    # Add arguments to the parser
    for k in task_config:
        parser.add_argument(f"--{k}")
    args = parser.parse_args()
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}
    task_config.update(args_dict)

    # Run the main function
    main(task_config)
