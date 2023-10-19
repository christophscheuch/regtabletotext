from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    name='regtabletotext',
    version='0.0.4',
    description='Helpers to print regression output as well-formated text',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(),
    author='Christoph Scheuch',
    author_email='christoph.scheuch@gmail.com',
    keywords=['Regression', 'Table', 'Formatting', 'Quarto', 'Text'],
    url='https://github.com/christophscheuch/regtabletotext',
    download_url='https://pypi.org/project/regtabletotext/'
)

install_requires = [
    'statsmodels',
    'linearmodels'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
