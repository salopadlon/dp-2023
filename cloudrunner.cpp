#include "cloudrunner.hpp"
#include "ui_cloudrunner.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <jsoncpp/json/json.h>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <pcl/registration/icp.h>
#include <pcl/common/common.h>

CloudRunner::CloudRunner (QWidget *parent) :
  QMainWindow (parent),
  ui (new Ui::CloudRunner)
{
  ui->setupUi (this);
  this->setWindowTitle ("Cloudrunner");

  // Set up the QVTK window
  viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
  ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
  viewer->setupInteractor (ui->qvtkWidget->GetInteractor (), ui->qvtkWidget->GetRenderWindow ());
  ui->qvtkWidget->update ();

  // Seed the random number generator with the current time
  srand(time(nullptr));

  // Load file | Works with PLY files
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud_filtered (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud2 (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud3 (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud4 (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud2_down (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud3_down (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud4_down (new pcl::PointCloud<pcl::PointXYZ> ());

  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();

  loadPointCloud("./../data/CL360global.ply", global_cloud_filtered);

  // loadPointCloud("./../data/first_new_icp.ply", local_cloud2);
  loadPointCloud("./../data/second_new_icp.ply", local_cloud3);
  // loadPointCloud("./../data/third_new_icp.ply", local_cloud4);

  // // loadPointCloud("./../data/first_new_downsampled_icp.ply", local_cloud2_down);
  // loadPointCloud("./../data/second_new_downsampled_icp.ply", local_cloud3);
  // loadPointCloud("./../data/third_new_downsampled_icp.ply", local_cloud4_down);

  // icp.setInputSource(local_cloud2);
  // icp.setInputTarget(global_cloud_filtered);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud2(new pcl::PointCloud<pcl::PointXYZ>);
  // icp.align(*aligned_cloud2);

  drawPointCloud(ui, viewer, global_cloud_filtered, "global_cloud_filtered");

  // // get the min/max bounds of the cloud
  // pcl::PointXYZ global_cloud_min, global_cloud_max;
  // pcl::getMinMax3D(*global_cloud_filtered, global_cloud_min, global_cloud_max);

  // // print the min/max bounds
  // std::cout << "Minimum x: " << global_cloud_min.x << std::endl;
  // std::cout << "Minimum y: " << global_cloud_min.y << std::endl;
  // std::cout << "Minimum z: " << global_cloud_min.z << std::endl;
  // std::cout << "Maximum x: " << global_cloud_max.x << std::endl;
  // std::cout << "Maximum y: " << global_cloud_max.y << std::endl;
  // std::cout << "Maximum z: " << global_cloud_max.z << std::endl;

  // get the min/max bounds of the cloud
  pcl::PointXYZ local_cloud_min, local_cloud_max;
  pcl::getMinMax3D(*local_cloud3, local_cloud_min, local_cloud_max);

  // print the min/max bounds
  std::cout << "Minimum x local: " << local_cloud_min.x << std::endl;
  std::cout << "Minimum y local: " << local_cloud_min.y << std::endl;
  std::cout << "Minimum z local: " << local_cloud_min.z << std::endl;
  std::cout << "Maximum x local: " << local_cloud_max.x << std::endl;
  std::cout << "Maximum y local: " << local_cloud_max.y << std::endl;
  std::cout << "Maximum z local: " << local_cloud_max.z << std::endl;

  // Create a KD tree for searching nearest neighbors
  kdtree->setInputCloud(global_cloud_filtered);

  // Initialize Monte Carlo Localization parameters
  float y_deviation = 1.0;
  double sigma = 0.05;

  // Calculate the rmse_reference
  double rmse_reference6 = calculateRMSE(kdtree, local_cloud3);
  std::cout << "Reference RMSE: " << rmse_reference6 << endl;

  // Define the range of x and y coordinates within which the local point cloud can be placed
  // double x_min = global_cloud_min.x + local_cloud_size.x() / 2.0;
  // double x_max = global_cloud_max.x - local_cloud_size.x() / 2.0;
  // double y_min = (global_cloud_min.y + local_cloud_size.y()) / 2.0;
  // double y_max = (global_cloud_max.y - local_cloud_size.y()) / 2.0;

  double x_min = -60;
  double x_max = 60;
  double y_min = -60;
  double y_max = 100;

  std::random_device rd;
  std::mt19937 generator(rd());
  
  std::normal_distribution<double> x_distribution_gauss_robot(0.0, 0.03f);
  std::normal_distribution<double> y_distribution_gauss_robot(0.0, y_deviation);

  std::uniform_real_distribution<double> x_distribution(x_min, x_max);
  std::uniform_real_distribution<double> y_distribution(y_min, y_max);

  // // std::cout << "x_distribution: " << x_distribution << endl;
  // // std::cout << "y_distribution: " << y_distribution << endl;

  // // std::cout << "gauss_distribution: " << distribution_gauss_robot << endl;

  int parameter = 0;
  int num_particles = 10;
  int total_particles = 30;

  std::map<double, std::array<double, 4>> rmseCoords;
  std::vector<double> x_coords(num_particles);
  std::vector<double> y_coords(num_particles);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> particles(num_particles);
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> new_particles;
  int it_weight = 0;

  // Generate random x and y coordinates within the defined range
  // #pragma omp parallel for 
  for (int it = 0; it < 10; it++) {
  for (int i = 0; i < num_particles; ++i) {
    // UNIFORM
    // x_coords[i] = x_distribution(generator);
    // y_coords[i] = y_distribution(generator);

    // GAUSS
    x_coords[i] = x_distribution_gauss_robot(generator);
    y_coords[i] = y_distribution_gauss_robot(generator);
  }
  }

  omp_lock_t lock;
  omp_init_lock(&lock);

  // Initialize particle set with random poses
  // #pragma omp parallel for 
  for (int it = 0; it < num_particles; it++) {       
    pcl::PointCloud<pcl::PointXYZ>::Ptr particle(new pcl::PointCloud<pcl::PointXYZ>);
    transform.translation() << x_coords[it], y_coords[it], 0.0f; // set z coordinate to desired value
    pcl::transformPointCloud(*local_cloud3, *particle, transform);
    particles[it] = particle;

    // Calculate rmse
    double rmse = calculateRMSE(kdtree, particle);

    std::array<double, 4> coords = {x_coords[it], y_coords[it], 0.0, 0.0};

    omp_set_lock(&lock);
    rmseCoords.insert(std::make_pair(rmse, coords));
    omp_unset_lock(&lock);
  }

  omp_destroy_lock(&lock);

  // Create a vector of pairs to store RMSE values and their coordinates
  std::vector<std::pair<double, std::array<double, 4>>> rmseVector;

  // Copy the contents of rmseCoords to rmseVector
  for (const auto& entry : rmseCoords)
  {
      rmseVector.emplace_back(entry.first, entry.second);
  }

  // Sort rmseVector by the RMSE values in descending order
  std::sort(rmseVector.begin(), rmseVector.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
  });

  // Get the first 10 pairs in rmseVector
  int numAreas = 0.2*num_particles;
  std::vector<std::pair<double, std::array<double, 4>>> bestAreas(rmseVector.begin(), rmseVector.begin() + numAreas);

  // Compute normalized weights
  std::vector<double> normalized_weights;
  double total_weight = std::accumulate(bestAreas.begin(), bestAreas.end(), 0.0,[](double sum, const std::pair<double, std::array<double, 4>>& pair) {
      return sum + pair.first;
  });

  for (const auto& w : bestAreas) {
      normalized_weights.push_back(w.first / total_weight);
  }

  for (const auto& area : bestAreas) {
    // Generate a random sample from the Gaussian distribution
    std::normal_distribution<double> x_distribution_gauss(area.second[0], sigma);
    std::normal_distribution<double> y_distribution_gauss(area.second[1], sigma);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> particles_new;
    int num_particles2 = normalized_weights[it_weight]*total_particles;

    for (int i = 0; i < num_particles2; ++i) {
      // std::cout << "x_distribution: " << x_distribution_gauss << endl;
      // std::cout << "y_distribution: " << y_distribution_gauss << endl;

      // std::normal_distribution<double> z_distribution(mean.z(), sigma);
      double x_particle = x_distribution_gauss(generator);
      double y_particle = y_distribution_gauss(generator);

      // Generate a new particle at the random sample location
      pcl::PointCloud<pcl::PointXYZ>::Ptr particle_new(new pcl::PointCloud<pcl::PointXYZ>);
      transform.translation() << x_particle, y_particle, 0.0f;
      pcl::transformPointCloud(*local_cloud3, *particle_new, transform);
      particles_new.push_back(particle_new);

      // Calculate rmse
      double rmse = calculateRMSE(kdtree, particle_new);

      std::array<double, 4> coords = {x_particle, y_particle, 0.0, 0.0};
      rmseCoords.insert(std::make_pair(rmse, coords));
    }

    it_weight++;

    // Add the generated particles to the resampled particles vector
    new_particles.insert(new_particles.end(), particles_new.begin(), particles_new.end());
  }

  auto lowest_rmse = rmseCoords.begin(); // iterator to the first key-value pair
  double rmse = lowest_rmse->first; // rmse value
  std::array<double, 4> coords_min = lowest_rmse->second; // corresponding coordinates

  std::cout << rmse << "," << num_particles << "," << coords_min[0] << "," << coords_min[1] << "," << coords_min[2] << "," << coords_min[3] << "," << std::endl;

  transform.translation() << coords_min[0], coords_min[1], coords_min[2];
  std::cout << transform.matrix() << std::endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud (*local_cloud3, *transformed_cloud, transform);

  for (int j = 0; j < num_particles; j++) {
    std::stringstream ss;
    ss << "particle_" << j;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> particle_color_handler (particles[j], 20, 20, 230); // Blue
    viewer->addPointCloud(particles[j], particle_color_handler, ss.str());
  }

  for (int j = 0; j < new_particles.size(); j++) {
    std::stringstream ss;
    ss << "new_particle_" << j;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> particle_color_handler (new_particles[j], 20, 230, 230); // Cyan
    viewer->addPointCloud(new_particles[j], particle_color_handler, ss.str());
  }

  // Define R,G,B colors for the point cloud
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> local_cloud2_color_handler (local_cloud3, 230, 20, 20); // Red
  viewer->addPointCloud (local_cloud3, local_cloud2_color_handler, "original_cloud");
  //   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> local_cloud3_color_handler (aligned_cloud3, 20, 230, 20); // Red
  // viewer->addPointCloud (aligned_cloud3, local_cloud3_color_handler, "original_cloud2");
  //   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> local_cloud4_color_handler (aligned_cloud4, 20, 20, 230); // Red
  // viewer->addPointCloud (aligned_cloud4, local_cloud4_color_handler, "original_cloud3");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 20, 230, 20); // Green
  viewer->addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");
}

CloudRunner::~CloudRunner ()
{
  delete ui;
}

void CloudRunner::loadPointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  // Check file extension
  std::string extension = filename.substr(filename.find_last_of(".") + 1);

  if (extension == "pcd") {
      if (pcl::io::loadPCDFile(filename, *cloud) < 0) {
          std::cerr << "Failed to load PCD file: " << filename << std::endl;
      }
  }
  else if (extension == "ply") {
      if (pcl::io::loadPLYFile(filename, *cloud) < 0) {
          std::cerr << "Failed to load PLY file: " << filename << std::endl;
      }
  }
  else {
      std::cerr << "Unsupported file extension: " << extension << std::endl;
  }
}

void CloudRunner::loadPointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  // Check file extension
  std::string extension = filename.substr(filename.find_last_of(".") + 1);

  if (extension == "pcd") {
      if (pcl::io::loadPCDFile(filename, *cloud) < 0) {
          std::cerr << "Failed to load PCD file: " << filename << std::endl;
      }
  }
  else if (extension == "ply") {
      if (pcl::io::loadPLYFile(filename, *cloud) < 0) {
          std::cerr << "Failed to load PLY file: " << filename << std::endl;
      }
  }
  else {
      std::cerr << "Unsupported file extension: " << extension << std::endl;
  }
}

void CloudRunner::drawPointCloud(Ui::CloudRunner* ui, pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, char const* name)
{
  viewer->addPointCloud (cloud, name);
  viewer->resetCamera ();
  ui->qvtkWidget->update ();
}

void CloudRunner::drawPointCloud(Ui::CloudRunner* ui, pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, char const* name)
{
  viewer->addPointCloud (cloud, name);
  viewer->resetCamera ();
  ui->qvtkWidget->update ();
}

void CloudRunner::removeGround(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud_filtered)
{
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 100.0);
  // pass.setNegative (true);
  pass.filter (*cloud_filtered);
}

void CloudRunner::removeOutliers(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud_filtered)
{
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.setNegative (true);
  sor.filter (*cloud_filtered);
}

float CloudRunner::getMinJsonKey(Json::Value json)
{
  // Find the lowest value of key in the JSON object
  float min_key = std::numeric_limits<float>::max();

  for (auto const& key : json.getMemberNames()) {
      float f_key = std::stof(key);
      if (f_key < min_key) {
          min_key = f_key;
      }
  }

  return min_key;
}

double CloudRunner::calculateRMSE(pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree, pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud)
{
  double sum_squared_distance = 0.0;

  // #pragma omp parallel for reduction(+:sum_squared_distance)
  for (int i = 0; i < local_cloud->points.size(); i++)
  {
      std::vector<int> indices(1);
      std::vector<float> distances(1);
      if (kdtree->nearestKSearch(local_cloud->points[i], 1, indices, distances) > 0)
      {
          sum_squared_distance += distances[0] * distances[0];
      }
  }
  
  // Calculate the average distance
  double rmse_distance = sqrt(sum_squared_distance / local_cloud->points.size());
  return rmse_distance;
}

float CloudRunner::generateRandomNumber()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);  // Generate a random number in the range from 0.01 to 0.5
  float random_num = dist(gen);
  return random_num;
}