/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cstdlib>

#include "helper_functions.h"
#include "map.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::normal_distribution;
using std::default_random_engine;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  gen.seed((unsigned)std::chrono::system_clock::now().time_since_epoch().count());
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_th(theta, std[2]);

  num_particles = 100;  // TODO: Set the number of particles

  for (int k=0; k < num_particles; ++k) {
    particles.push_back(Particle{k, dist_x(gen), dist_y(gen), dist_th(gen), 1});
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  for (Particle &p : particles) {
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_th(0, std_pos[2]);

    double noise_x = dist_x(gen);
    double noise_y = dist_y(gen);

    if (yaw_rate != 0) {
      p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
    }
    else {
      p.x += velocity * cos(p.theta) * delta_t;
      p.y += velocity * sin(p.theta) * delta_t;
    }

    p.x += noise_x;
    p.y += noise_y;

    p.theta += yaw_rate*delta_t + dist_th(gen);
    // normalize angle
    p.theta = p.theta - int(p.theta/(2 * M_PI)) * (2 * M_PI);
//    while (x_diff(3)> M_PI) x_diff(3)-=2*M_PI;
//    while (x_diff(3)<-M_PI) x_diff(3)+=2*M_PI;
  }
}

vector<int> ParticleFilter::dataAssociation(const Map &map_landmarks, vector<LandmarkObs> &observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    vector<int> assoc;

    // Iterate through each transformed observation to associate to a landmark
    for (LandmarkObs obs : observations) {
        int closest_landmark = -1;
        double min_dist = 999999;
        double curr_dist;
        // Iterate through all landmarks to check which is closest
        for (int j = 0; j < int(map_landmarks.landmark_list.size()); ++j) {
          Map::single_landmark_s lm = map_landmarks.landmark_list[j];
          // Calculate Euclidean distance
          curr_dist = sqrt(pow(obs.x - lm.x_f, 2)
                           + pow(obs.y - lm.y_f, 2));
          // Compare to min_dist and update if closest
          if (curr_dist < min_dist) {
            min_dist = curr_dist;
            closest_landmark = j;
          }
        }

        assoc.push_back(closest_landmark);
    }

  return assoc;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (Particle &particle : particles) {
    vector<LandmarkObs> trans_obs;
    for (LandmarkObs obs : observations) {
      LandmarkObs t_obs = transform_obs(particle, obs);
      // reduce number of observations per particle due to the limited sensor_range
      if (sensor_range >= dist(particle.x, particle.y, t_obs.x, t_obs.y))
        trans_obs.push_back(t_obs);
    }

    vector<int> landmark_assoc = dataAssociation(map_landmarks, trans_obs);

    double std_dev_obs_x = std_landmark[0];
    double std_dev_obs_y = std_landmark[1];
    particle.weight = 1.;
    for (int i=0; i<int(landmark_assoc.size()); ++i) {
      int lm = landmark_assoc[i];
      Map::single_landmark_s landmark = map_landmarks.landmark_list[lm];
      LandmarkObs obs = trans_obs[i];

      double prob = 1/(2*M_PI*std_dev_obs_x*std_dev_obs_y) * exp(-(pow(obs.x-landmark.x_f,2)/(2*pow(std_dev_obs_x,2))
                                                            + (pow(obs.y-landmark.y_f,2)/(2*pow(std_dev_obs_y,2)))));
      particle.weight *= prob;
    }
  }
}

LandmarkObs ParticleFilter::transform_obs(const Particle &p, const LandmarkObs &obs) {
  // Transform the x and y coordinates
  double x_map, y_map;
  x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
  y_map = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);

  // Create new Position to hold transformed observation
  LandmarkObs transformed_obs;
  transformed_obs.id = obs.id;
  transformed_obs.x = x_map;
  transformed_obs.y = y_map;

  return transformed_obs;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());


  vector<Particle> resampled_particles;

  vector<double> weights;
  for (Particle& p : particles) {
    weights.push_back(p.weight);
  }

  std::discrete_distribution<> d(weights.begin(), weights.end());

  for (int k=0; k<num_particles; ++k) {
    int p = d(gen);
    Particle rp = particles[p];
    resampled_particles.push_back(rp);
  }

  this->particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}