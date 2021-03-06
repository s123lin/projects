from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

#################################
# Zip Code within Travis County #
#################################
search_text = "78759"

####################
# File output name #
####################
file_output_name = "property_id.txt"

########################################################
# Access Travis County Website and Search for Zip Code #
########################################################
driver = webdriver.Chrome()
driver.get("http://propaccess.traviscad.org/clientdb/?cid=1")
elem = driver.find_element_by_name("propertySearchOptions:searchText")
elem.clear()
elem.send_keys(search_text)
elem.send_keys(Keys.RETURN)

#############################################
# Iterate through all pages in the site and #
# extract property ID's from page source.   #
#############################################
current_URL = driver.current_url
append_URL = "?rtype=address&page="
soup = BeautifulSoup(driver.page_source, 'lxml')
base_url = "SearchResults.aspx?rtype=address&page="
length_base = len(base_url)
total_pages = 1;
	
for link in soup.find_all('a'):
	getURL = link.get('href') + ""
	if getURL[0] == "S":
		number = getURL[length_base:]
		if int(number) > total_pages:
			total_pages = int(number)
			
f = open(file_output_name,"w+")

for data in soup.find_all('span'):
	extract = data.get('prop_id')
	if extract != None:
		f.write(str(extract) + "\n")
		
for i in range(2,total_pages + 1):
	driver.get(current_URL + append_URL + str(i))
	soupy = BeautifulSoup(driver.page_source, 'lxml')
	
	for data in soupy.find_all('span'):
		extract = data.get('prop_id')
		if extract != None:
			f.write(str(extract) + "\n")
		
f.close