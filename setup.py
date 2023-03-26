import setuptools

setuptools.setup(
    name='ClassiPyGRB',
    version='0.0.1',
    author='Garcia-Cifuentes, K., Becerra, R. L., De Colle, F.',
    author_email='keneth.garcia@correo.nucleares.unam.mx',
    description='Package to classify and visualize GRBs from Swift based on Manifold Learning Algortihms.',
    long_description='ClassiPyGRB is a novel package created to perform non-supervised Machine Learning Algorithms on '
                     'Gamma-Ray-Bursts (GRB) data from Swift. This package emphasizes performing automatic classifications with '
                     'several applications, such as transient identification and support in telescope follow-up.',
    long_description_content_type='text/markdown',
    url='https://github.com/KenethGarcia/ClassiPyGRB',
    python_requires='>=3.8',
    install_requires=['scipy', 'numpy', 'scikit-learn', 'pandas', 'requests', 'matplotlib', 'moviepy', 'tqdm', 'fabada', ''],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Education',
                 'Intended Audience :: Information Technology',
                 'Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: Implementation',
                 'Natural Language :: English',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX :: Linux',
                 ],
    package_dir={'': 'ClassiPyGRB'},
    packages=setuptools.find_packages(where='ClassiPyGRB'),
)
