Welcome to the **Text Classification AI** repository! This project provides an AI-powered system for analyzing and categorizing textual data into predefined categories. Follow this guide to configure, install, and use the system.

---

## Features
- Flexible categorization based on customizable configurations.
- Works with any dataset containing text data.
- Uses pre-defined or user-defined categories for classification.
- Supports one-hot or categorical output encoding.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/161tochtli/text-classification-ai.git
cd text-classification-ai
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Configuration

The system relies on a configuration YAML file to define input, output, and processing parameters. Modify the provided `task_configuration.yaml` to match your use case.

### Example Configuration
```yaml
# Path to the input file
input_file: data/input_file.csv

# Path to the output file
output_file: data/output_file.csv

# Parameter columns
param_cols:
  - comentario

# Fixed parameters
fixed_params:
  idioma: Español
  categories:
    - Positivo
    - Negativo

# Reprocess flag
reprocess: false

# Encoding
one_hot_encoding: false

# Model name
model: gpt-4o-mini

# Batch size
batch_size: 32
```

### What to Customize
1. **`input_file`**: Path to the file containing text data to classify.
2. **`output_file`**: Path where the classified data will be saved.
3. **`param_cols`**: Columns in the input file to process (e.g., `comentario`).
4. **`fixed_params`**: Define fixed parameters and categories for classification.
5. **`one_hot_encoding`**: Set to `true` for one-hot encoding output; `false` for categorical labels.
6. **`model`**: Specify the model to use (e.g., `gpt-4o-mini`).
7. **`batch_size`**: Number of rows to process per batch.

---

## Usage

### Run the Script
To classify text, use the following command:

```bash
python tasks/data_classification.py --task sample_task
```

### Options
- `--task`: Name of the task to perform. It should be the name of the folder inside the `tasks` directory, containing the task configuration file and prompt files.

---

## Output

The output file (defined in the `output_file` parameter) will contain the input data along with the classification results. Example:

| comentario                       | classification |
|----------------------------------|----------------|
| "This product is amazing!" | Positive       |
| "I didn't like the service."      | Negative       |

---

## Project Structure
```
.secrets/               # Directory for sensitive credentials (e.g., API keys)
.venv/                  # Virtual environment (optional)
data/                   # Directory for input and output files
  input_file.csv        # Input dataset
  output_file.csv       # Output results

tasks/                  # Directory for tasks and configurations
  sample_task/          # Example task configuration
    task_configuration.yaml  # Task configuration file
    user_prompt.txt     # File containing the user prompt template
    system_prompt.txt   # File containing the system prompt template

.config                 # Global configuration file, here you set working directory

requirements.txt        # Python dependencies
```

---

## Support
For questions or issues, please create a GitHub issue or contact me.
