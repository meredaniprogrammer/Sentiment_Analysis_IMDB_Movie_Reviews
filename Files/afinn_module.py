# afinn_module.py

from afinn import Afinn

def initialize_afinn():
    """
    Initializes and returns the AFINN sentiment analysis tool.
    """
    afinn = Afinn()
    return afinn

def calculate_afinn_score(review, afinn_tool):
    """
    Calculates and returns the AFINN score for a given review.
    :param review: The review text.
    :param afinn_tool: Initialized AFINN tool.
    :return: AFINN score for the review.
    """
    return afinn_tool.score(review)

def compute_afinn_scores(reviews):
    """
    Compute AFINN scores for a list of reviews.
    :param reviews: List of reviews.
    :return: List of AFINN scores.
    """
    afinn_tool = initialize_afinn()
    scores = [calculate_afinn_score(review, afinn_tool) for review in reviews]
    return scores
