from bs4 import BeautifulSoup as bs4
import requests


class Scrapper():
    """"Class to scrape reviews"""

    def __init__(self, url):
        self.url = url
        self.reviews = {'Review' : [], 'Label': []} 

    def __get_urls(self, html):
        """Get urls from the website"""
        websites = html.find_all('a', class_='search-result-heading')
        if websites:
            return [f'https://www.trustpilot.com/{i["href"]}' for i in websites]
        # else:
        #     # print(html.find_all('div', class_='property-card'))
        #     test = html.find_all('div', class_='property-card')[0]
        #     print(test)
        #     websites = [i['href'] for i in html.find_all('a')]
        #     print(websites)

    def __get_html(self, url):
        """Get the html form the url"""

        # initialize a session
        session = requests.Session()

        # set the User-agent as a regular browser
        session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"

        html = session.get(url).content

        #parse the response HTML page
        page_html = bs4(html, 'html.parser')

        return page_html

    def __get_reviews(self, page_html):
        """Get all reviews from the page"""
        return page_html.find_all(class_='review__content') or page_html.find_all(class_='review-card-container')

    def __get_text(self, review_html):
        """Get the content of the review"""
        if review_html.find('p', class_='review-content__text'):
            return review_html.find('p', class_='review-content__text').text.strip()
        else:
            return review_html.find('div', class_='content').text.strip()

    def __get_label(self, review_html):
        """Get the label of the review"""
        if review_html.find('img'):
            return 'Positive' if int(review_html.find('img')['alt'][0]) >= 3 else 'Negative'
        else:
            return 'Positive' if int(review_html.find('div', class_='score orange small').text.strip()) >= 5.5 else 'Negative'

    def __reviews_loop(self, reviews_list):
        """Loop trhough the reviews on the page"""
        for review in reviews_list:
            try:
                text = self.__get_text(review)
                label = self.__get_label(review)
                self.reviews['Review'].append(text)
                self.reviews['Label'].append(label)
            except Exception as e:
                print(e)

    def get_reviews(self):
        urls_list_html = self.__get_html(self.url)
        urls = self.__get_urls(urls_list_html)

        for url in urls:
            html = self.__get_html(url)
            reviews = self.__get_reviews(html)
            self.__reviews_loop(reviews)
        return self.reviews

    def hostel_world(self):
        # urls_list_html = self.__get_html(self.url)
        # # with open(f"pagehtml.html", "w") as file:
        # #     file.write(str(urls_list_html))
        # #     file.close()

        # # print(urls_list_html)
        # urls = self.__get_urls(urls_list_html)
        # print(urls)
        html = self.__get_html(self.url)
        reviews = self.__get_reviews(html)
        self.__reviews_loop(reviews)
        return self.reviews


# """ THIS IS JUST FOR TESTRING """
# # final = Scrapper('https://www.trustpilot.com/search?query=hotels').get_reviews()
# # print(final)
# # THIS IS JUST FOR TESTRING
# final = Scrapper('https://www.hostelworld.com/pwa/hosteldetails.php/Queen-Elizabeth-Chelsea/London/50443?from=2021-09-21&to=2021-09-24&guests=1&display=reviews').hostel_world()
# # final = Scrapper('https://www.hostelworld.com/s?q=London,%20England&country=England&city=London&type=city&id=3&from=2021-09-21&to=2021-09-24&guests=1&page=1').hostel_world()
# print(final) 
# for i in range(len(final['Review'])):
#     print(final['Review'][i] + ', ' + final['Label'][i])
# # [print(final.get(i)) for i in final['Review']]
# # [print(i + ', ' + final[i] + '\n' for i in final['Review'])] 
# """ End Test """

# # url_list = ['https://www.hostelworld.com/pwa/hosteldetails.php/Wombat-s-The-City-Hostel-London/London/88047?from=2021-09-21&to=2021-09-24&guests=1&display=reviews',
# # ]