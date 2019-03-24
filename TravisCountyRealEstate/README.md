# Travis County Real Estate Search

**Purpose**: The goal is to implement a program that automatically extracts information on tens/hundreds of thousands of properties in order to statistically evaluate property worth and to aid in purchasing decisions.

**Packages/Dependencies**: Python 3.5, Selenium, BeautifulSoup, Chrome Web Browser

**Current Status**: The python script _travis_county_search.py_ employs Selenium web driver to aid in navigating the [Travis County Records Website](http://propaccess.traviscad.org/clientdb/?cid=1) in order to extract property ID's of properties in a specific zip code (line 8). The property ID's are important due to the fact that the website uses these property ID's in the URL's of each unique property. By extracting this information, we can iterate through each property's informational site and extract further information on the property. The program outputs a file called _property_id.txt_ (line 13) that has all property ID's for designated zip code.

**Future Work**: 
- Determining what property information is most correlated with property value (e.g. for statistical analysis) as well as usefulness (e.g. communication with the property owners)
- Parsing the information from page source into JSON data
- Using ML to correlate property attributes with overall value; incorporate real estate API like Zillow
