"""Contains various function used throughout this project.

Attributes:
    cache_dir (TYPE): Description
    log_dir (TYPE): Description
    logger (TYPE): Description
    output_dir (TYPE): Description
    root_dir (TYPE): Description
"""
# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from typing import Dict
import os
import json
import pickle
import sys
import pycountry
from custom_logger import CustomLogger
import subprocess
import smtplib
from email.message import EmailMessage

root_dir = os.path.dirname(__file__)
cache_dir = os.path.join(root_dir, '_cache')
log_dir = os.path.join(root_dir, '_logs')
output_dir = os.path.join(root_dir, '_output')

logger = CustomLogger(__name__)  # use custom logger


def get_secrets(entry_name: str, secret_file_name: str = 'secret') -> Dict[str, str]:
    """
    Open the secrets file and return the requested entry.

    Args:
        entry_name (str): Description
        secret_file_name (str, optional): Description

    Returns:
        Dict[str, str]: Description
    """
    with open(os.path.join(root_dir, secret_file_name)) as f:
        return json.load(f)[entry_name]


_config_cache = None


def _load_json_file(file_name: str):
    """
    Load a JSON file from the repository root.

    Args:
        file_name (str): Name of the file relative to the repository root.

    Returns:
        dict or None: Parsed content, or None if the file does not exist.

    Raises:
        json.decoder.JSONDecodeError: If the file is badly formatted.
    """
    try:
        with open(os.path.join(root_dir, file_name)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_configs(entry_name: str, config_file_name: str = 'config', config_default_file_name: str = 'default.config'):
    """
    Open the config file and return the requested entry.
    Values in config override default.config. If config does not exist, or does not
    contain the entry, the value from default.config is used.

    Args:
        entry_name (str): Name of the entry.
        config_file_name (str, optional): Name of the user config file.
        config_default_file_name (str, optional): Name of the default config file.

    Returns:
        TYPE: Value of the entry.

    Raises:
        KeyError: If the entry is in neither config nor default.config.
    """
    global _config_cache
    if _config_cache is None:
        # check if config file is valid
        if not check_config(config_file_name, config_default_file_name):
            sys.exit(1)
        content = dict(_load_json_file(config_default_file_name))
        content.update(_load_json_file(config_file_name) or {})
        _config_cache = content
    return _config_cache[entry_name]


def check_config(config_file_name: str = 'config',
                 config_default_file_name: str = 'default.config'):
    """
    Check that default.config exists and that both config files are valid JSON.
    Warn about entries of default.config missing in config, which fall back to their defaults.

    Args:
        config_file_name (str, optional): Name of the user config file.
        config_default_file_name (str, optional): Name of the default config file.

    Returns:
        bool: True if the configuration can be used.
    """
    # load default.config file
    try:
        default = _load_json_file(config_default_file_name)
    except json.decoder.JSONDecodeError:
        logger.error('Default config file {} badly formatted.', config_default_file_name)
        return False
    if default is None:
        logger.error('Default config file {} not found.', config_default_file_name)
        return False
    # load config file
    try:
        config = _load_json_file(config_file_name)
    except json.decoder.JSONDecodeError:
        logger.error('Config file {} badly formatted. Please update based on {}.',
                     config_file_name,
                     config_default_file_name)
        return False
    if config is None:
        logger.info('Config file {} not found, using {}.', config_file_name, config_default_file_name)
        return True
    # report entries that are missing in config
    missing = [key for key in default if key not in config]
    if missing:
        logger.warning('Config file {} is missing {}. Using values from {}.',
                       config_file_name,
                       ', '.join(missing),
                       config_default_file_name)
    return True


def search_dict(dictionary, search_for, nested=False):
    """
    Search if dictionary value contains certain string search_for. If
    nested=True multiple levels are traversed.

    Args:
        dictionary (dict): Dict to search in.
        search_for (str): What to search for.
        nested (bool, optional): If dictionary nested or not.

    Returns:
        str: Description.
    """
    for k in dictionary:
        if nested:
            for v in dictionary[k]:
                if search_for in v:
                    return k
                elif v in search_for:
                    return k
        else:
            if search_for in dictionary[k]:
                return k
            elif dictionary[k] in search_for:
                return k
    return None


def save_to_p(file, data, description_data='data'):
    """
    Save data to a pickle file.

    Args:
        file (str): Pickle file (*.p or *.pkl).
        data (tuple): Data tuple.
        description_data (str, optional): Description of data.
    """
    path = os.path.join(os.path.join(root_dir, 'trust'), file)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info('Saved ' + description_data + ' to pickle file {}.', file)


def load_from_p(file, description_data='data'):
    """Load data from a pickle file.

    Args:
        file (str): Pickle file (*.p or *.pkl).
        description_data (str, optional): Description of data.

    Returns:
        tuple: data tuple.
    """
    path = os.path.join(os.path.join(root_dir, 'trust'), file)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    logger.info('Loaded ' + description_data + ' from pickle file {}.',
                file)
    return data


def correct_country(country):
    """
    Corrects common country name variations for compatibility with pycountry.countries.get(name=...).

    Args:
        country (str): Name of country in its full form.

    Returns:
        str: Corrected country.
    """
    corrections = {
        'Russia': 'Russian Federation',
        'Syria': 'Syrian Arab Republic',
        'South Korea': 'Korea, Republic of',
        'North Korea': "Korea, Democratic People's Republic of",
        'Korea': 'Korea, Republic of',
        'Iran': 'Iran, Islamic Republic of',
        'Vietnam': 'Viet Nam',
        'Venezuela': 'Venezuela, Bolivarian Republic of',
        'Bolivia': 'Bolivia, Plurinational State of',
        'Moldova': 'Moldova, Republic of',
        'Laos': "Lao People's Democratic Republic",
        'Brunei': 'Brunei Darussalam',
        'Czech Republic': 'Czechia',
        'Ivory Coast': "Côte d'Ivoire",
        'Cape Verde': 'Cabo Verde',
        'Swaziland': 'Eswatini',
        'Macau': 'Macao',
        'Taiwan': 'Taiwan, Province of China',
        'Tanzania': 'Tanzania, United Republic of',
        # 'United States': 'United States of America',
        'UK': 'United Kingdom',
        'Palestine': 'Palestine, State of',
        'Micronesia': 'Micronesia, Federated States of',
        'Bahamas': 'Bahamas, The',
        # 'Gambia': 'Gambia, The',
        'São Tomé and Príncipe': 'Sao Tome and Principe',
        'Turkiye': 'Turkey',
        'Türkiye': 'Turkey',
        'Congo (Democratic Republic)': 'Congo, The Democratic Republic of the',
        'Congo (Congo-Brazzaville)': 'Congo',
        'Burma': 'Myanmar',
        'East Timor': 'Timor-Leste',
        'Saint Kitts': 'Saint Kitts and Nevis',
        'Saint Vincent': 'Saint Vincent and the Grenadines',
        'Saint Lucia': 'Saint Lucia',
        'Antigua': 'Antigua and Barbuda',
        'Trinidad': 'Trinidad and Tobago',
        'Slovak Republic': 'Slovakia',
        'Vatican': 'Holy See',
    }

    return corrections.get(country, country)


# Convert ISO-3 to country name
def iso3_to_country_name(iso3):
    """
    Get ISO-3 code for a country passed as ISO-3.

    Args:
        iso3 (str): ISO-3 code of a country.

    Returns:
        TYPE: ISO-3 code.
    """
    try:
        country = pycountry.countries.get(alpha_3=iso3.upper())
        return country.name if country else None
    except KeyError:
        return None


# Fetch ISO-2 country data
def get_iso2_country_code(country_name):
    """
    Get ISO-2 code for a country passed as its full name.

    Args:
        country_name (str): Full name of a country.

    Returns:
        TYPE: ISO-2 code.
    """
    if country_name == 'Kosovo':
        return 'XK'
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_2  # ISO-2 code
        else:
            return "Country not found"
    except KeyError:
        return "Country not found"


# Fetch ISO-3 country data
def get_iso3_country_code(country_name):
    """
    Get ISO-3 code for a country passed as its full name.

    Args:
        country_name (str): Full name of a country.

    Returns:
        TYPE: ISO-3 code.
    """
    if country_name == 'Kosovo':
        return 'XKX'
    try:
        country = pycountry.countries.get(name=country_name)
        print(country_name)
        if country:
            return country.alpha_3  # ISO-3 code
        else:
            return "Country not found"
    except KeyError:
        return "Country not found"


# Pull changes from repository
def git_pull():
    """
    git pull changes from the repo with a terminal command.
    """
    try:
        logger.info("Attempting to pull latest changes from git repository...")
        result = subprocess.run(["git", "pull"], capture_output=True, text=True, check=True)
        logger.info(f"Git pull successful:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git pull failed with error:\n{e.stderr}")


# Send email with certain message
def send_email(subject, content, sender, recipients):
    """
    Send email with certain subject and content from sender to recipients.

    Args:
        subject (str): Subject of email.
        content (str): Content of email.
        sender (str): Email address to send from.
        recipients (list): Email addresses to send to.
    """
    msg = EmailMessage()
    msg.set_content(content)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    # Try to send email
    try:
        with smtplib.SMTP_SSL(get_secrets("email_smtp"), 465) as smtp:
            smtp.login(get_secrets("email_account"), get_secrets("email_password"))
            smtp.send_message(msg)
            logger.info(f"Sent email to: {recipients}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
