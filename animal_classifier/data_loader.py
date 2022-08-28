from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import download_images, resize_images, verify_images, get_image_files
from pathlib import Path
from time import sleep


class DDGImageLoader(object):

    def __init__(self,
                 terms: list):
        self.terms = terms

    def load(self) -> dict:
        """
        :return: term_url_map: Dictionary mapping search term to a list of URLs
        """
        term_url_map = self.get_term_url_map(self.terms)
        return term_url_map

    def save(self,
             url_map: dict,
             dest: str):
        """
        :param url_map: Dictionary mapping search term to a list of URLs
        :param dest: Path to save directory as a string
        """
        dest_root = Path(dest)
        for term, urls in url_map.items():
            dest_path = (dest_root / term)
            dest_path.mkdir(exist_ok=True, parents=True)

            # download images into destination
            print(f'Downloading images for search term "{term}"')
            download_images(dest_path, urls=urls)

            # delete failed downloads
            failed = verify_images(get_image_files(dest_path))
            failed.map(Path.unlink)

            # pause to avoid server overload
            sleep(5)

    def get_term_url_map(self,
                         terms: list) -> dict:
        """
        :param terms: List of search terms
        :return: term_url_map: Dictionray mapping search term to a list of URLs
        """

        term_url_map = {}

        for term in terms:
            urls = self.get_urls(term)
            term_url_map[term] = urls

        return term_url_map

    @staticmethod
    def get_urls(term: str,
                 max_images=30) -> list:
        """
        :param term: Term to search for as a string
        :param max_images: Maximum number of images to retrieve
        :return: List of URLs
        """
        print(f'Searching for "{term}"')
        return L(ddg_images(term, max_results=max_images)).itemgot('image')


if __name__ == '__main__':
    TERMS = ['dog', 'cat']
    DEST = 'animal_pictures'

    # load urls
    loader = DDGImageLoader(terms=TERMS)
    url_map = loader.load()

    # save to destination
    loader.save(url_map, dest=DEST)
