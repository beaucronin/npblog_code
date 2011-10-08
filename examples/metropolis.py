"""
An example of Metropolis sampling for probabilistic inference.
"""

import random
import math

def bivariate_normal(x, mu, sigma, rho):
	"""
	The bivariate normal distribution; see 
	http://en.wikipedia.org/wiki/Multivariate_normal_distribution
	"""
	return (1. / (2. * math.pi * sigma[0] * sigma[1] * math.sqrt(1. - rho * rho))) * \
	  math.exp( (-1. / (2. * (1. - rho * rho))) * \
	    ( ( math.pow(x[0] - mu[0], 2) / sigma[0] * sigma[0] ) + \
	      ( math.pow(x[1] - mu[1], 2) / sigma[1] * sigma[1] ) - \
	      ( 2. * rho * (x[0] - mu[0]) * (x[1] - mu[1]) / (sigma[0] * sigma[1]) ) ) )

def prob(x):
	"""
	Define the distribution over which we will be performing inference. In a 
	real-world setting, this may be a high-dimensional object that is very hard
	to characterize or sum over.

	For this demo, we will use the sum of two 2D normal distributions.
	"""
	return 1.   * bivariate_normal(x, (0., 1.2), (1., 1.), .8) + \
	       1.05 * bivariate_normal(x, (.6, -1.), (1.3, .7), -.6)
	
def propose(x, jump = 0.1):
	"""
	Propose a new location to jump to by making a normally-distributed step
	from the current location.
	"""
	return (x[0] + random.gauss(0, jump), x[1] + random.gauss(0, jump))

def print_result(accept, old_prob, new_prob, x):
	if accept:
		prefix = "* ACCEPT"
	else:
		prefix = "  REJECT"
	print ("%(prefix)s move to (%(x0)6.3f, %(x1)6.3f), " + \
				 "p = %(p)7.5f (%(dp)7.5f)") % \
				 { 
					 'prefix': prefix,
					 'x0': x[0], 
					 'x1': x[1], 
					 'p': new_prob, 
					 'dp': min(1.0, new_prob / old_prob)
				 }

class MetropolisDemo():
	def __init__(self, steps = 100):
		self.history = []
		self.x = (0., 0.)

	def run(self, steps = 100, jump_size = 0.5):
		current_prob = prob(self.x)
		for i in range(steps):
			# Generate a proposal and compute its probability
			x_star = propose(self.x, jump_size)
			prob_star = prob(x_star)
			
			# Decide whether to accept the proposal
			u = random.random()
			if prob_star > current_prob or u < prob_star / current_prob:
				print_result(True, current_prob, prob_star, x_star)
				self.x = x_star
				current_prob = prob_star
			else:
				print_result(False, current_prob, prob_star, x_star)
			self.history.append(self.x)
		return self.history


