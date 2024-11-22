# uv venv --python 3.12
# uv pip install torch
# uv sync --no-build-isolation

conda create -n ai-hack python=3.12
conda activate ai-hack

conda install numpy pandas scikit-learn pytorch torchvision -c defaults -c pytorch -c conda-forge
pip install torch-geometric torch-cluster torch-scatter mesh-reducer

