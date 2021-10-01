#!/usr/bin/env python3
import requests
import argparse
from sys import exc_info, stderr, stdout
from os import makedirs
from os.path import dirname, join as pathjoin
import logging
from urllib.parse import urljoin, urlsplit, urlunsplit
from reppy.robots import Robots
from reppy.exceptions import ReppyException
from bs4 import SoupStrainer, BeautifulSoup as Soup
import json
import configparser


MAX_DEPTH = 1
CHUNK_SIZE = 128

CACHE = {}

USER_AGENT = 'Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                      'Chrome/60.0.3112.113 YaBrowser/17.9.1.888 Yowser/2.5 Safari/537.36 '
CONTENT_TYPE = 'text/html'

LOG_FILE = './log/scrapper.log'
LOG_PATTERN = '"[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"'
LOG_LEVEL = logging.INFO


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.config:
        global_config(load_config(args.config))

    init_logger(LOG_FILE, LOG_PATTERN, LOG_LEVEL)
    logging.info('Started')

    headers = {
        'User-Agent': USER_AGENT,
        'Content-Type': CONTENT_TYPE,
    }

    out_dir = args.pages_dir

    site = ''
    try:
        site = fix_site(args.site)
    except ValueError:
        err = exc_info()[1]
        logging.critical(err)
        parser.error(err)

    pages = ()
    if '*' in args.pages:
        pages = get_pages_from_sitemaps(site, headers=headers)
        if len(pages) == 0:
            pages = recursive_get_links(site, headers=headers)

    try:
        pages = fix_pages(pages or args.pages)
    except ValueError:
        err = exc_info()[1]
        logging.critical(err)
        parser.error(err)

    makedirs(out_dir, exist_ok=True)
    try:
        save_pages(site, pages, out_dir, headers=headers)
    except PermissionError:
        err = 'You are do not have enough permissions'
        logging.critical(err)
        exit(2)

    logging.info('Success')


def global_config(config):
    global USER_AGENT, CONTENT_TYPE, MAX_DEPTH, CHUNK_SIZE, LOG_FILE, LOG_LEVEL, LOG_PATTERN
    USER_AGENT = config.http.user_agent or USER_AGENT
    CONTENT_TYPE = config.http.content_type or CONTENT_TYPE
    MAX_DEPTH = config.system.max_recursion_depth or MAX_DEPTH
    CHUNK_SIZE = config.system.chunk_size or CHUNK_SIZE
    LOG_LEVEL = config.log.level or LOG_LEVEL
    LOG_PATTERN = config.log.pattern or LOG_PATTERN
    LOG_FILE = config.log.file or LOG_FILE


def load_config(config_file):
    if config_file.endswith('.ini'):
        return load_ini_config(config_file)
    elif config_file.endswith('.json'):
        return load_json_config(config_file)
    else:
        print('Wrong config file format', file=stderr)
        exit(3)


def load_json_config(config_file):
    try:
        with open(config_file, 'r') as config:
            return json.loads(config)
    except json.decoder.JSONDecodeError:
        print('Wrong config file format', file=stderr)
        exit(3)


def load_ini_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config.values()


def save_pages(site, pages, out_dir, **kwargs):
    logging.info('Saving pages...')
    i = 0
    for page in pages:
        i += 1
        logging.info('Saving page: %s', page)
        url = urljoin(site, page)
        with open(pathjoin(out_dir, str(i) + '.json'), 'wb') as output_file:
            try:
                r = CACHE.get(url) or requests.get(url, **kwargs)
                r.raise_for_status()
                if 'Content-Type' in r.headers and not r.headers['Content-Type'].startswith(CONTENT_TYPE):
                    raise requests.RequestException()
                CACHE[url] = r
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    output_file.write(chunk)
            except requests.RequestException:
                logging.warning('Can not get access to: %s', page)
                continue


def fix_site(site):
    scheme, netloc, path, query, params = urlsplit(site)
    if scheme == '':
        scheme = 'http'
    if netloc == '':
        netloc, path = path, ''
    if netloc == '':
        raise ValueError('Wrong URI: %s' % site)
    return urlunsplit((scheme, netloc, path, query, params))


def fix_pages(pages):
    result = []
    for page in pages:
        scheme, netloc, path, query, params = urlsplit(page)
        if netloc != '':
            if scheme == '':
                scheme = 'http'
            result.append(urlunsplit((scheme, netloc, path, query, params)))
        elif path == '':
            raise ValueError('Wrong URI: %s' % page)
        else:
            result.append(path)
    return result


def get_pages_from_sitemaps(url, **kwargs):
    sitemaps = get_sitemaps(url)
    result = ()
    logging.info('Trying to get list of all pages from sitemaps')
    for sitemap in sitemaps:
        logging.info('Loading sitemap from: %s', sitemap)
        try:
            r = CACHE[sitemap] or requests.get(sitemap, **kwargs)
            r.raise_for_status()
            if 'Content-Type' in r.headers and not (
                        r.headers['Content-Type'].startswith('text/xml') or
                        r.headers['Content-Type'].startswith('application/xml')
            ):
                raise requests.RequestException()
            CACHE[sitemap] = r
        except requests.RequestException:
            logging.warning('Can not get access to: %s', sitemap)
            continue
        soup = Soup(r.content, 'xml')
        result += soup('loc')

    if len(result) == 0:
        return recursive_get_links(url, **kwargs)
    return result


FORBIDDEN_PREFIXES = ('#', 'tel:', 'mailto:', 'javascript:')


def recursive_get_links(site, **kwargs):
    logging.warning('Trying to get list of all links by parsing all site')
    links = set()
    add_all_links_recursive(site, links, **kwargs)
    return links


def add_all_links_recursive(site, links, maxdepth=MAX_DEPTH, **kwargs):
    links_to_handle_recursive = []
    try:
        logging.info('Loading page: %s', site)
        r = CACHE.get(site) or requests.get(site, **kwargs)
        r.raise_for_status()
        if 'Content-Type' in r.headers and not r.headers['Content-Type'].startswith(CONTENT_TYPE):
            raise requests.RequestException()
        CACHE[site] = r
    except requests.RequestException:
        logging.warning('Can not get access to: %s', site)
        return
    soup = Soup(r.content, 'lxml', parse_only=SoupStrainer('a'))
    for tag_a in soup('a'):
        a = tag_a['href']
        if all(not a.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            if a.startswith('/') and not a.startswith('//'):
                a = urljoin(site, a)
            if urlsplit(a).netloc == urlsplit(site).netloc and urlsplit(a).path not in links:
                links.add(urlsplit(a).path)
                links_to_handle_recursive.append(a)
    if maxdepth > 0:
        for link in links_to_handle_recursive:
            add_all_links_recursive(link, links, maxdepth - 1)


def get_sitemaps(url):
    logging.info('Trying to get sitemaps from robots.txt')
    robots_url = urljoin(url, '/robots.txt')
    try:
        return list(Robots.fetch(robots_url).sitemaps)
    except ReppyException:
        logging.warning('Can not get access to: %s', robots_url)
        return [urljoin(url, 'sitemap.xml')]


def init_logger(file, fmt, level):
    if file == 'stderr':
        file = stderr
    elif file == 'stdout':
        file = stdout
    else:
        makedirs(dirname(file), exist_ok=True)
    logging.basicConfig(filename=file, format=fmt, level=level)


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Load specified data from pages from specified site')
    parser.add_argument('--pages_dir', '-d', metavar='DIR', required=True,
                        help='Directory to store loaded pages')
    parser.add_argument('--site', '-s', required=True,
                        help='''URI of site from witch to load pages''')
    parser.add_argument('--pages', '-p', metavar='PAGE', nargs='+', default='*',
                        help='Relative paths to pages to load and parse')
    parser.add_argument('--config', '-c',
                        help='''Read all params from specified config file''')
    return parser


if __name__ == '__main__':
    main()
