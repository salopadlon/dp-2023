#pragma once

#include <iostream>

// Qt
#include <QMainWindow>

// Point Cloud Library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>

// Visualization Toolkit (VTK)
#include <vtkRenderWindow.h>

// JSON library
#include <jsoncpp/json/json.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui
{
  class CloudRunner;
}

class CloudRunner : public QMainWindow
{
  Q_OBJECT

public:
  explicit CloudRunner (QWidget *parent = 0);
  ~CloudRunner ();

  void loadPointCloud(const std::string&, pcl::PointCloud<pcl::PointXYZRGB>::Ptr);
  void loadPointCloud(const std::string&, pcl::PointCloud<pcl::PointXYZ>::Ptr);

  void drawPointCloud(Ui::CloudRunner*, pcl::visualization::PCLVisualizer::Ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr, char const*);
  void drawPointCloud(Ui::CloudRunner*, pcl::visualization::PCLVisualizer::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr, char const*);

  void removeGround(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>);
  void removeOutliers(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>, std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>);

protected:
  pcl::visualization::PCLVisualizer::Ptr viewer;
  PointCloudT::Ptr cloud;

private:
  Ui::CloudRunner *ui;

  float getMinJsonKey(Json::Value);
  double calculateRMSE(pcl::search::KdTree<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr);
  float generateRandomNumber();

};
 
