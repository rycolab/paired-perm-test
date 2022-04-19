import setuptools

setuptools.setup(
    name = 'pairedpermtest',
    version = '1.0',
    author = 'Ran Zmigrod, Tim Vieira, Ryan Cotterell',
    author_email = '',
    description = 'Exact p-values for paired-permutation tests',
    long_description = '',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/rycolab/paired-perm-test/',
    packages = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[],
)
