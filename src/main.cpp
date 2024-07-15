/*
 * Copyright (c) 2020, Robobrain.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/* Author: Konstantinos Konstantinidis */

#include "ros/ros.h"
#include "datmo.hpp"
#include <string>

bool compareVectorSize(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() < b.size(); // Ordine inverso per ottenere il vettore più grande
}
/*
    ALL'AUMENTARE DELLE PERFORMANCE DIMINUISCE IL TEMPO DI ESECUZIONE
*/
int main(int argc, char **argv)
{
   //creo il file dove salvo i tempi
   std::ofstream of;
   of.open("/mnt/c/Users/tolve/OneDrive/Desktop/Tirocinio/tempo_per_iterazione.csv");
   if (of.is_open())
    of << "CallBack;Clustering;transformPointList;visualiseGroupedPoints;MsgPub;Cluster::Update;newCluster;rectangleFitting;cluster_cuda;newLshapeTracker;LshapeUpdate\n";
   of.close();

  //Initiate ROS
  ros::init(argc, argv, "datmo_node");

  //Create an object of class datmo 
  Datmo  datmo_object;

  auto start = std::chrono::high_resolution_clock::now();
  ros::spin();
  auto endOpt = std::chrono::high_resolution_clock::now();
  memFree();
  std::cout << "total time: " << std::chrono::duration_cast<std::chrono::microseconds>(endOpt - start).count() << std::endl;
  std::cout << "il nodo si attiva " << datmo_object.callBack.size() - 1 << std::endl;
  std::vector<std::vector<double>> tempi_totali; //contenitore per tutti i vettori così da scorrerli più comodamente
  std::vector<double> avg;
  //calcolo media tempo di esecuzione
  avg.push_back(datmo_object.callBack[0] / datmo_object.callBack.size() - 1);
  avg.push_back(datmo_object.clustering[0] / datmo_object.clustering.size() - 1);
  avg.push_back(datmo_object.transform[0] / datmo_object.transform.size() - 1);
  avg.push_back(datmo_object.visualise[0] / datmo_object.visualise.size() - 1);
  avg.push_back(datmo_object.msgpub[0] / datmo_object.msgpub.size() - 1);
  avg.push_back(Cluster::clusterupd[0] / Cluster::clusterupd.size() - 1);
  avg.push_back(Cluster::newcl[0] / Cluster::newcl.size() - 1);
  avg.push_back(Cluster::retfit[0] / Cluster::retfit.size() - 1);
  avg.push_back(Cluster::closecrit[0] / Cluster::closecrit.size() - 1);
  avg.push_back(LshapeTracker::lshape[0] / LshapeTracker::lshape.size() - 1);
  avg.push_back(LshapeTracker::lshapeupdate[0] / LshapeTracker::lshapeupdate.size() - 1);

  //inserimento dei vettori
  tempi_totali.push_back(datmo_object.callBack);
  tempi_totali.push_back(datmo_object.clustering);
  tempi_totali.push_back(datmo_object.transform);
  tempi_totali.push_back(datmo_object.visualise);
  tempi_totali.push_back(datmo_object.msgpub);
  tempi_totali.push_back(Cluster::clusterupd);
  tempi_totali.push_back(Cluster::newcl);
  tempi_totali.push_back(Cluster::retfit);
  tempi_totali.push_back(Cluster::closecrit);
  tempi_totali.push_back(LshapeTracker::lshape);
  tempi_totali.push_back(LshapeTracker::lshapeupdate);

  //trovo l'array di dimensione maggiore
  auto largestVector = std::max_element(tempi_totali.begin(), tempi_totali.end(), compareVectorSize);
  //scrivo il file csv
  std::ofstream file("/mnt/c/Users/tolve/OneDrive/Desktop/Tirocinio/prestazioni.csv");
  if (file.is_open()) {
      file << ";CallBack;Clustering;transformPointList;visualiseGroupedPoints;MsgPub;Cluster::Update;newCluster;rectangleFitting;cluster_cuda;newLshapeTracker;LshapeUpdate\nMedia;";
      for (auto media : avg) file << media << ";";
      file << "\nsommaTempi;";
      for (int i = 0; i < (*largestVector).size(); i++) {
          for (auto array : tempi_totali) {
              file << ((i < array.size()) ? (int) array[i] : 0) << ";";
          }
          file << "\n;";
      }
      file.close();
  }
  return 0;
}

