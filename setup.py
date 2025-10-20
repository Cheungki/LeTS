import os
from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
with open(os.path.join(version_folder, 'src/version')) as f:
    __version__ = f.read().strip()

install_requires = [
  # verl
  'accelerate',
  'codetiming',
  'datasets',
  'dill',
  'hydra-core',
  'numpy',
  'pandas',
  'peft',
  'pyarrow>=15.0.0',
  'pybind11',
  'pylatexenc',
  'ray[default]>=2.10',
  'tensordict==0.6.2',
  'torchdata',
  'transformers',
  'vllm==0.8.4',
  'wandb',

  # flashrag
  'datasets',
  'base58',
  'nltk',
  'numpy',
  'langid',
  'openai',
  'peft',
  'PyYAML',
  'rank_bm25',
  'rouge',
  'spacy',
  'tiktoken',
  'torch',
  'tqdm',
  'transformers>=4.40.0',
  'bm25s[core]==0.2.0',
  'fschat',
  'streamlit',
  'chonkie>=0.4.0',
  'gradio>=5.0.0',
  'rouge-chinese',
  'jieba',

  # others
  'sglang',
  'jsonlines',
]

setup(
    name='lets',
    version=__version__,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    # url='',
    license='MIT License',
    author='Zhejiang University',
    author_email='cheung_se@zju.edu.cn',
    description='LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization',
    install_requires=install_requires,
    package_data={'': ['**/*.yaml']},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)