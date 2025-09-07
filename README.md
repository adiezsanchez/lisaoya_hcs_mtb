# lisaoya_hcs_mtb
Processing of 2D fluorescence multichannel images derived from a HCS assay. Acquired in a Nikon Crestoptics V3 Spinning Disk.

Environment creation command

<code>mamba create -n hcs_cellpose python=3.10 napari cellpose pyclesperanto-prototype apoc-backend pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=12.1 plotly python-kaleido nd2 ipykernel ipython -c conda-forge -c pytorch -c nvidia</code>
<code>pip install spotiflow</code>

Still testing venv WIP

<h1>New pixi section (WIP)</h1>

<img src="./assets/pixi_banner.svg">

This assumes you have Git installed, if not follow the instructions in this video:

1. Open you cmd, copy the following command and hit enter. This will install Pixi.

<code>powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"</code>

2. Close your cmd, reopen it, copy the following command.

git clone https://github.com/adiezsanchez/lisaoya_hcs_mtb && cd lisaoya_hcs_mtb && pixi install

3. Now you have a working environment to run the scripts contained in this repository.