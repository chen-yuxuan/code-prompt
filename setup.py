from setuptools import setup

setup(
    name="code-prompt",
    packages=["code_prompt"],
    include_package_data=True,
    version="1.0.0",
    description="Benchmarking of Code-LLMs for NLP Classification",
    author="Yuxuan Chen",
    author_email="yuxuan.chen@dfki.de",
    install_requires=[
        "numpy==1.26.4",
        # "torch",
        "transformers",
        "datasets>=4.0.0",
        "huggingface-hub",
        "protobuf>=3.20",
        "scikit-learn",
        "sentencepiece",
        "tqdm",
        "vllm",
        "openai",
        "google",
    ],
)
