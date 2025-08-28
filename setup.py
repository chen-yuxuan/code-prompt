from setuptools import setup, find_packages

setup(
    name="code-prompt",
    packages=["code_prompt"],
    include_package_data=True,
    version="1.0.0",
    description="Benchmarking of Code-LLMs for NLP Classification",
    author="Yuxuan Chen",
    author_email="yuxuan.chen@dfki.de",
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "huggingface-hub",
        "numpy",
        "openai",
        "protobuf==3.20",
        "scikit-learn",
        "sentencepiece",
        "tqdm",
        "vllm",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
    ],
)
