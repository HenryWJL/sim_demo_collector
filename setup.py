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
        "fpsample==0.3.3",
        "gymnasium==1.0.0",
        "hydra-core==1.3.2",
        "matplotlib==3.10.6",
        "numpy==1.25.0",
        "omegaconf==2.3.0",
        "open3d==0.19.0",
        "robomimic==0.2.0",
        "robosuite==1.4.0",
        "scipy==1.15.1",
        "tqdm==4.67.1",
        "zarr==2.17.0",
    ]
)