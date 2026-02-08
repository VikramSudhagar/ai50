import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability = {}
    links = corpus[page]
    num_of_links = len(links)
    num_of_pages = len(corpus)
    probability_per_link = 0
    
    if num_of_links > 0:
        probability_per_link = damping_factor / num_of_links
    
    remain = (1 - damping_factor) / num_of_pages
    
    probability[page] = remain
    
    for key in links:
        probability[key] = probability_per_link + remain
    return probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    page_count = {}
    for keys in corpus:
        page_count[keys] = 0
    
    # Choose random starting page
    pages = list(corpus.keys())
    current_page = random.choice(pages)
    random_cnt = 0
    choice_cnt = 0
    
    for _ in range(n):
        page_count[current_page] += 1
        choice = random.uniform(0,1)
        
        if choice <= damping_factor:
            choice_cnt += 1
            prob_distribution = transition_model(corpus, current_page, damping_factor)
            choose_link_from_page = random.uniform(0,1)
            
            running_prob = 0
            for page, probability in prob_distribution.items():
                running_prob += probability
                if choose_link_from_page <= running_prob:
                    current_page = page
                    break
        else:
            random_cnt += 1
            current_page = random.choice(pages)            
    
    probability_distribution = {}
    
    for key, value in page_count.items():
        probability_distribution[key] = value/n
    return probability_distribution
    
    
    

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    probability_page = {}
    N = len(corpus)
    
    for key in corpus:
        probability_page[key] = 1/N
    
    # Repeat until convergence
    cnt = 0
    while cnt < 100:
        new_prob_page = {}
        for key in corpus:
            cur_page = key
            # Start
            part1 = (1 - damping_factor)/N
            running_probability = 0
            for key in corpus:
                if key == cur_page:
                    continue
                page_links_set = corpus[key]
                if cur_page not in page_links_set:
                    continue
                
                if len(page_links_set):
                    running_probability += damping_factor * (probability_page[key]/len(page_links_set))
                else:
                    running_probability += damping_factor * (probability_page[key]/N)
            prev_prob = probability_page[cur_page]
            new_prob_page[cur_page] = part1 + running_probability
            dx = abs(new_prob_page[cur_page] - prev_prob)
            if dx < 0.001:
                return probability_page
        probability_page = new_prob_page
        cnt += 1


if __name__ == "__main__":
    main()
