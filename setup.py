from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name = "sim_demo_collector",
    version=__version__,
    description="Expert demonstration collector in simulation",
    author="Junlin Wang",
    author_email="wangjl@seas.upenn.edu",
    url="https://github.com/HenryWJL/sim_demo_collector/",
    packages=[
        package for package in find_packages() 
        if package.startswith("sim_demo_collector")
    ],
    python_requires=">=3.10",
    setup_requires=["setuptools>=62.3.0"],
    include_package_data=True,
    install_requires=[
        "zarr==2.17.0",
        "numpy==1.26.4",
        "scipy==1.15.1",
        "torch==2.2.0",
        "torchvision==0.17.0",
        "diffusers==0.27.2",
        "tqdm==4.67.1",
        "hydra-core==1.2.0",
        "omegaconf==2.3.0",
        "einops==0.8.0",
        "numba==0.61.0",
        "gymnasium==1.0.0",
        "robosuite==1.4.0",
        "robomimic==0.2.0",
    ]
)