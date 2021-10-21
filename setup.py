import setuptools

setuptools.setup(
    name='heatmap_digitizer',
    version='0.1.0',
    url='https://github.com/gilmarGNJ/heatmap-digitizer',
    license='GNU General Public License v3.0',
    author='gilmarGNJ',
    author_email='gilmar_jr@outlook.com',
    description='Python code to extract numerical data from heatmap images.',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pandas", "matplotlib", "seaborn", "opencv-python", "scipy"],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'': ['data/*.png']},
)
