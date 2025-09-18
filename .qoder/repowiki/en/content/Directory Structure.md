# Directory Structure

<cite>
**Referenced Files in This Document**   
- [main.py](file://main.py)
- [bartending_agent.py](file://bartending_agent.py)
- [requirements.txt](file://requirements.txt)
- [notebooks/gradio_ui_testing.ipynb](file://notebooks/gradio_ui_testing.ipynb)
- [notebooks/mvp_notebook_kaggle.ipynb](file://notebooks/mvp_notebook_kaggle.ipynb)
- [notebooks/submission_notebook.ipynb](file://notebooks/submission_notebook.ipynb)
- [README.md](file://README.md)
</cite>

## Table of Contents
1. [Directory Structure](#directory-structure)
2. [Root-Level Files](#root-level-files)
3. [Notebooks Directory](#notebooks-directory)
4. [Development and Production Separation](#development-and-production-separation)
5. [Recommended Workflow](#recommended-workflow)

## Root-Level Files

The root directory of the MayaBartendingAgent project contains essential files that define the application's entry point, core logic, and dependencies. These files form the backbone of the production-ready application.

### main.py

The `main.py` file serves as the primary entry point for the application. It is responsible for launching the Gradio-based user interface and orchestrating the interaction between the UI components and the core business logic implemented in `bartending_agent.py`. The file imports necessary functions such as `process_order` and `get_voice_audio` from the `bartending_agent` module, ensuring a clean separation between the interface and the logic. It configures logging for debugging and operational visibility and defines the Gradio UI layout, including components like the chatbot, text input, audio output, and avatar image. The UI is built using Gradio's Blocks interface, allowing for a flexible and responsive design. Event handlers are set up to manage user input submission and conversation clearing, with session state managed through Gradio's `State` components to maintain conversation history and order details across interactions. The application is launched in debug mode for local development, making it easy to iterate and test changes.

**Section sources**
- [main.py](file://main.py#L1-L143)

### bartending_agent.py

The `bartending_agent.py` file contains the core business logic of the bartending agent. This module is designed to be stateless, with functions that process user input, manage the drink menu, and generate responses using the Gemini LLM. It handles API interactions with both Google's Gemini service for text generation and Cartesia for text-to-speech synthesis. The module includes robust error handling and retry mechanisms using the `tenacity` library to ensure reliability when calling external APIs. The menu is defined as a static dictionary, and functions like `get_menu_text` generate formatted text for display. The `process_order` function constructs a prompt for the LLM by combining the current conversation history, order state, and menu information, then processes the LLM's response to update the order based on heuristic detection of drink additions. The `get_voice_audio` function synthesizes the agent's response into audio using Cartesia, with special handling to pronounce the bar's name "MOK 5-ha" as "Moksha" for cultural accuracy. This file is imported by `main.py` and provides all the backend functionality required by the Gradio interface.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L1-L375)

### requirements.txt

The `requirements.txt` file specifies the Python dependencies required to run the application. It lists the exact versions or minimum versions of libraries that are necessary for the project, ensuring consistent environments across different development and deployment setups. Key dependencies include `google-generativeai` for interacting with the Gemini API, `gradio` for the web interface, `cartesia` for text-to-speech functionality, `tenacity` for retry logic, and `python-dotenv` for loading environment variables. This file is used during the setup process to install all required packages in a virtual environment, as documented in the README. It is crucial for both local development and deployment to platforms like Kaggle, where the environment must be reproducible.

**Section sources**
- [requirements.txt](file://requirements.txt#L1-L9)

## Notebooks Directory

The `notebooks/` directory houses a collection of Jupyter notebooks used for development, testing, and competition submission. These notebooks are instrumental during the iterative development phase, allowing for rapid prototyping and experimentation with different features and configurations.

### gradio_ui_testing.ipynb

The `gradio_ui_testing.ipynb` notebook is dedicated to testing the Gradio user interface within the notebook environment. It allows developers to validate the UI components, layout, and interaction flows without running the full `main.py` application. This notebook contains code to install dependencies, set up logging, and initialize the Gemini and Cartesia APIs. It implements a self-contained version of the Gradio interface with the same core logic as the production application, enabling developers to test changes to the UI design, such as column layouts, component styling, and event handling. The notebook is particularly useful for ensuring that the UI functions correctly in a notebook context, which is important for demonstration and sharing purposes on platforms like Google Colab.

**Section sources**
- [notebooks/gradio_ui_testing.ipynb](file://notebooks/gradio_ui_testing.ipynb#L1-L573)

### MVP Notebook Variants

The directory contains several variants of the MVP (Minimum Viable Product) notebook, including `mvp_notebook_kaggle.ipynb`, `mvp_notebook_kaggle_compatible.ipynb`, `mvp_notebook_kaggle_compliance.ipynb`, and `mvp_notebook_kaggle_merged.ipynb`. These notebooks represent different stages and versions of the development process, likely reflecting iterative improvements and adaptations for the Kaggle competition. They contain comprehensive implementations of the bartending agent with additional features such as Retrieval-Augmented Generation (RAG) using FAISS for vector search, function calling with LangChain, and detailed setup instructions for API keys. These notebooks serve as a sandbox for developing and testing advanced features before they are integrated into the production codebase. The presence of multiple variants suggests a process of experimentation and refinement, with the `merged` version likely representing a consolidated implementation.

**Section sources**
- [notebooks/mvp_notebook_kaggle.ipynb](file://notebooks/mvp_notebook_kaggle.ipynb#L1-L2467)
- [notebooks/mvp_notebook_kaggle_compatible.ipynb](file://notebooks/mvp_notebook_kaggle_compatible.ipynb)
- [notebooks/mvp_notebook_kaggle_compliance.ipynb](file://notebooks/mvp_notebook_kaggle_compliance.ipynb)
- [notebooks/mvp_notebook_kaggle_merged.ipynb](file://notebooks/mvp_notebook_kaggle_merged.ipynb)

### Submission Notebooks

The `submission_notebook.ipynb` and its associated Python script `submission_notebook.py` are specifically designed for submission to the Kaggle competition. These files contain the final, competition-ready version of the code, optimized for the Kaggle environment. They include specific logic for retrieving API keys from Kaggle secrets, which is a requirement for running the notebook in the Kaggle platform. The notebook also contains detailed documentation on how to set up the necessary API keys for Gemini and Cartesia, making it a self-contained package for submission and evaluation. The `kaggle_test.py` file appears to be a utility script that encapsulates the API key setup and LLM initialization logic used in the submission notebook, promoting code reuse and consistency.

**Section sources**
- [notebooks/submission_notebook.ipynb](file://notebooks/submission_notebook.ipynb#L1-L2887)
- [kaggle_test.py](file://kaggle_test.py#L1-L123)

## Development and Production Separation

The project structure clearly separates development and experimental work from the production code. The `notebooks/` directory is dedicated to development, testing, and submission, containing various iterations and experimental versions of the agent. In contrast, the root-level Python files (`main.py`, `bartending_agent.py`) represent the stable, production-ready application. This separation allows developers to experiment freely in the notebooks without affecting the core application. Changes and improvements validated in the notebooks can be carefully integrated into the production code. The `requirements.txt` file ensures that both environments use the same dependencies, maintaining consistency. This structure supports a workflow where new features are prototyped in notebooks, tested thoroughly, and then migrated to the main application files for deployment.

## Recommended Workflow

For new feature development, start by creating or modifying a notebook in the `notebooks/` directory to prototype and test the idea. Use the MVP notebook variants as a starting point for complex features like RAG or advanced agent workflows. Once the feature is stable and tested, integrate the relevant code into `bartending_agent.py` for the core logic or `main.py` for UI changes. Ensure that any new dependencies are added to `requirements.txt`. Use the `gradio_ui_testing.ipynb` notebook to validate UI changes before updating the production interface. For Kaggle submissions, use the `submission_notebook.ipynb` as a template, ensuring that API keys are retrieved from Kaggle secrets and that the code is compliant with competition rules. Always test the final application by running `main.py` locally to verify that all components work together as expected.