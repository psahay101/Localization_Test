#include "ros/ros.h"
#include <gazebo_msgs/GetModelState.h>
#include <geometry_msgs/Twist.h>
#include <string>
#include "map.h"
#include "helper_functions.h"
#include "sensor_msgs/LaserScan.h"
#include <math.h>
#include "particle_filter.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <iostream>
#include <sstream>
#include <iterator>
#include <nav_msgs/Odometry.h>

using namespace std;



ros::Time current_time, last_time;


vector<LandmarkObs> obs;
bool IsLastTimeInitialized = false;
bool MapzInitialized = false;
bool PositionInitialized = false;
bool UseApproxTime = true;
Maps mapz;
ros::ServiceClient client;
ros::ServiceClient client1;
ros::Publisher odom_pub;

int num_cones=9;//TO BE CHANGED EVERYTIME
double delta_t = 0.01; // Time elapsed between measurements [sec]//PLEASE CHANGE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double sensor_range = 10.0; // Sensor range [m]   
double sigma_pos [3] = {0.3, 0.3, 0.01}; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]
double previous_velocity=0.0;
double previous_yawrate=0.0;
double sense_x, sense_y, sense_z, sense_theta;
double best_x, best_y, best_theta;
double qx, qy, qz, qw;


ParticleFilter pf;





//some extra functions.------------------------------------------------------------------------------------------------------------------------------------------------


struct Quaternion

{

    double w, x, y, z;

};


Quaternion ToQuaternion(double roll, double pitch, double yaw) 

{

   

    double cy = cos(yaw * 0.5);

    double sy = sin(yaw * 0.5);

    double cp = cos(pitch * 0.5);

    double sp = sin(pitch * 0.5);

    double cr = cos(roll * 0.5);

    double sr = sin(roll * 0.5);

    

    Quaternion q;

    q.w = cy * cp * cr + sy * sp * sr;

    q.x = cy * cp * sr - sy * sp * cr;

    q.y = sy * cp * sr + cy * sp * cr;

    q.z = sy * cp * cr - cy * sp * sr;

    

    return q;

}



double ToEulerAngles(double x,double y, double z, double w)

{

    double yaw;

    double siny_cosp = +2.0 * (w * z + x * y);

    double cosy_cosp = +1.0 - 2.0 * (y * y + z * z);

    yaw = atan2(siny_cosp, cosy_cosp);

    

    return yaw;

}



// extra fns end ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






default_random_engine gen;


//for the update step
double bivariate_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
    return exp(-((x-mu_x)*(x-mu_x)/(2*sig_x*sig_x) + (y-mu_y)*(y-mu_y)/(2*sig_y*sig_y))) / (2.0*3.14159*sig_x*sig_y);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    num_particles = 20;
    weights.resize(num_particles);
    
    // init each particle
    for (int i = 0; i < num_particles; i++){
        Particle par;
        
        par.id = i;
        par.x = dist_x(gen);
        par.y = dist_y(gen);
        par.theta = dist_theta(gen);
        par.weight = 1;
        
        weights[i] = 1;
        
        particles.push_back(par);
        
    }
    
    is_initialized = true;
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    
    
    
    default_random_engine gen;
    
    int i;
    for (i = 0; i < num_particles; i++) {
        double particle_x = particles[i].x;
        double particle_y = particles[i].y;
        double particle_theta = particles[i].theta;
        
        double pred_x;
        double pred_y;
        double pred_theta;
        
        if (fabs(yaw_rate) < 0.0001) {
            pred_x = particle_x + velocity * cos(particle_theta) * delta_t;
            pred_y = particle_y + velocity * sin(particle_theta) * delta_t;
            pred_theta = particle_theta;
        } else {
            pred_x = particle_x + (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
            pred_y = particle_y + (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
            pred_theta = particle_theta + (yaw_rate * delta_t);
        }
        
        normal_distribution<double> dist_x(pred_x, std_pos[0]);
        normal_distribution<double> dist_y(pred_y, std_pos[1]);
        normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
        
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
    
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
    
    
    int i, j;
	for (i = 0; i < observations.size(); i++) {
	//Maximum distance can be square root of 2 times the range of sensor.
		double lowest_dist = sensor_range * sqrt(2);
		int closest_landmark_id = -1;
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;

		for (j = 0; j < predicted.size(); j++) {
		  double pred_x = predicted[j].x;
		  double pred_y = predicted[j].y;
		  int pred_id = predicted[j].id;
		  double current_dist = dist(obs_x, obs_y, pred_x, pred_y);

		  if (current_dist < lowest_dist) {
		    lowest_dist = current_dist;
		    closest_landmark_id = pred_id;
		  }
		}
		observations[i].id = closest_landmark_id;
	}
    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Maps map_landmarks) {
   
    
    weights.clear();
    
    // For all particles....
    
    for (int i=0; i<particles.size();i++){
        // Transforming observations from vehicle's co-ordinate system to map co-ordinate sytem
        vector<LandmarkObs> transformed_observations;
        
        for (int j=0; j<observations.size(); j++){
            LandmarkObs trans_obv;
            
            trans_obv.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
            trans_obv.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
            trans_obv.id = -1;
            
            transformed_observations.push_back(trans_obv);
        }
        
        // keeping only those landmarks which are in sensor range
        vector<LandmarkObs> predicted_landmarks;
        
        for (int j = 0; j <map_landmarks.landmark_list.size(); j++) {
            double landmark_dist;
            
            landmark_dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f );
            
            if (landmark_dist<sensor_range){
                LandmarkObs pred_landmark;
                pred_landmark.id = map_landmarks.landmark_list[j].id_i;
                pred_landmark.x = map_landmarks.landmark_list[j].x_f;
                pred_landmark.y = map_landmarks.landmark_list[j].y_f;
                
                predicted_landmarks.push_back(pred_landmark);
                
            }
        }
        
        
        dataAssociation(predicted_landmarks, transformed_observations, sensor_range);
        
        // Calculating the weights using multi variable gaissian probability
        double prob = 1;
        double mvgd;
        
        for (int j = 0; j < predicted_landmarks.size(); j++) {
            int id_min = -1;
            double min_dist = 99999;
            
            double px = predicted_landmarks[j].x;
            double py = predicted_landmarks[j].y;
            
            for (int k = 0; k < transformed_observations.size(); k++) {
                double tx = transformed_observations[k].x;
                double ty = transformed_observations[k].y;
                double curr_dist = dist(px, py, tx, ty);
                
                if (curr_dist< min_dist){
                    min_dist = curr_dist;
                    id_min = k;
                }
            }
            
            if (id_min != -1){
                mvgd = bivariate_normal(px, py, transformed_observations[id_min].x, transformed_observations[id_min].y, std_landmark[0], std_landmark[1]);
                
                prob = prob * mvgd;
            }
        }
        
        weights.push_back(prob);
        particles[i].weight = prob;
        
    }
    
    
}

void ParticleFilter::resample() {
    
    
    vector<Particle> resampled_particles;
    
    // Create a random particle generator.
    default_random_engine gen;
    
    //Generate random particle index.
    uniform_int_distribution<int> particle_index(0, num_particles - 1);
    
    int current_index = particle_index(gen);
    
    double beta = 0.0;
    
    double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());
    
    for (int i = 0; i < particles.size(); i++) {
        uniform_real_distribution<double> random_weight(0.0, max_weight_2);
        beta += random_weight(gen);
        
        while (beta > weights[current_index]) {
            beta -= weights[current_index];
            current_index = (current_index + 1) % num_particles;
        }
        resampled_particles.push_back(particles[current_index]);
    }
    particles = resampled_particles;
    
}

// code from here on was provided by udacity.

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    
    
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}










// ----------------------------------------------------------------------------------INTEGRATION CODE-------------------------------------------------------------------------------------------




















//Function to visualize the observations.
void showobs(vector<LandmarkObs> land){
	 for(int i=0;i<land.size();i++){
		 cout<<"ID is: "<<land[i].id<<endl;
		 cout<<"x is :"<<land[i].x<<endl;
		 cout<<"y is :"<<land[i].y<<endl;
		 cout<<endl;
	 }
}

//Function to visualize the landmarks.
void showdigs(vector<DistDegObs> digs){
	 for(int i=0;i<digs.size();i++){
		 cout<<"dist is :"<<digs[i].distance<<endl;
		 cout<<"angle is :"<<digs[i].degree<<endl;
		 cout<<endl;
	 }
}


void PublishOdom(){
	
	nav_msgs::Odometry odom;
	current_time = ros::Time::now();
	odom.header.stamp = current_time;
	odom.header.frame_id = "odom";
	odom.child_frame_id = "robot_footprint";
	odom.pose.pose.position.x = best_x;
	odom.pose.pose.position.y = best_y;
	odom.pose.pose.position.z = sense_z;
	odom.pose.pose.orientation.x=qx;
	odom.pose.pose.orientation.y=qy;
	odom.pose.pose.orientation.z=qz;
	odom.pose.pose.orientation.w=qw;
	odom.twist.twist.linear.x = previous_velocity;
	odom.twist.twist.linear.y = 0.0;
	odom.twist.twist.linear.z = 0.0;
	odom.twist.twist.angular.x = 0.0;
	odom.twist.twist.angular.y = 0.0;
	odom.twist.twist.angular.z = previous_yawrate;
	odom_pub.publish(odom);
	
}


void fillmapzs(Maps &m, int a, float b, float c){

    

    Maps::single_landmark_s si;

    si.id_i=a;

    si.x_f=b;

    si.y_f=c;

    

    m.landmark_list.push_back(si);

    

}

void getInitialPosition(){
	
	gazebo_msgs::GetModelState srv;
		
		string b="my_robot";
	    srv.request.model_name=b;

	    if(client1.call(srv)){
			sense_x=srv.response.pose.position.x;
			sense_y=srv.response.pose.position.y;
			sense_z=srv.response.pose.position.y;
			sense_theta=ToEulerAngles(srv.response.pose.orientation.x, srv.response.pose.orientation.y, srv.response.pose.orientation.z, srv.response.pose.orientation.w);
		}
	    else{
		ROS_WARN("service call to get_robot_state failed!");		
		}
	
	PositionInitialized = true;
	
}



void getLandmarks(){
	
	 gazebo_msgs::GetModelState srv;
	 
	int num=num_cones; // number of cones.	
    for(int i=0;i<num;i++){
		
		stringstream ss;
		ss << i;
		string s=ss.str();
		
		string b="Construction Cone_";
	    b=b+s;
	    srv.request.model_name=b;

	    if(client.call(srv)){
			fillmapzs(mapz, i, srv.response.pose.position.x, srv.response.pose.position.y);
			
		}
	    else{
		ROS_WARN("service call to get_model_state failed!");		
		}
	}
	MapzInitialized = true;
}


void runPf(){
	
	
	if(UseApproxTime ==true){
	
		delta_t=0.001;
		last_time=ros::Time::now();
		UseApproxTime=false;
	
	}
	else{
	
		current_time=ros::Time::now();
		delta_t= (current_time - last_time).toSec();
		last_time=ros::Time::now();
	
	}
	
	
	if (!pf.initialized()) {
		pf.init(sense_x, sense_y, sense_theta, sigma_pos);
	}
	else{
		pf.prediction(delta_t, sigma_pos, previous_velocity, previous_yawrate);//get vel and yawr from /cmd subscriber.
	}
	
	pf.updateWeights(sensor_range, sigma_landmark, obs, mapz);
	pf.resample();
	
	vector<Particle> particles = pf.particles;
	int num_particles = particles.size();
	double highest_weight = -1.0;
	Particle best_particle;
	double weight_sum = 0.0;
	for (int i = 0; i < num_particles; ++i) {
		if (particles[i].weight > highest_weight) {
			highest_weight = particles[i].weight;
			best_particle = particles[i];
		}
		weight_sum += particles[i].weight;
	}
	best_x=best_particle.x;
	best_y=best_particle.y;
	best_theta=best_particle.theta;
	
	Quaternion qq;
	qq=ToQuaternion(0.0, 0.0, best_theta);
	qx=qq.x;
	qy=qq.y;
	qz=qq.z;
	qw=qq.w;

}


void showmapz(Maps m){
	
	for(int k=0;k<m.landmark_list.size();k++){
	
		cout<<"MapsID is: "<<m.landmark_list[k].id_i<<endl;
		cout<<"Maps x_val is: "<<m.landmark_list[k].x_f<<endl;
		cout<<"Maps y_val is: "<<m.landmark_list[k].y_f<<endl;
		cout<<endl;
	}
	
}


void get_speed_callback(const geometry_msgs::Twist& msg){
	
	previous_velocity=msg.linear.x;
	previous_yawrate=msg.angular.z;
	
}




void get_laser_callback(const sensor_msgs::LaserScan scan)
{ 
	if(MapzInitialized == false){
		getLandmarks();
	}
	
	if(PositionInitialized == false){
		getInitialPosition();
	}
	
	obs.clear();
	float dist;
	float deg;
	float radians;
	double x,y; //coordiantes
	int id;
	int id1=0;
	int start=0;
	int end=scan.ranges.size();
	int count=0;
	int temp;
	LandmarkObs ob;
	DistDegObs dig;
	vector<DistDegObs> div;
	
	
	if((scan.ranges[0]<scan.range_max)&&(scan.ranges[end-1]<scan.range_max)){
		int j=0;
		while(scan.ranges[j]<scan.range_max){
			count++;
			j++;
		}
		start=j;
		j=end-1;
		while(scan.ranges[j]<scan.range_max){
			count--;
			j--;
		}
		end=j+1;
		if(count<0){
			id=scan.ranges.size()-1+(count/2);
		}
		else{
			id=count/2;
		}
		
		dist=scan.ranges[id];
		deg=id/2; //not robust as considers size of range to be 720.
		dig.distance=dist;
		dig.degree=deg;
		div.push_back(dig);
		
		
		
		radians=deg*(3.142/180.0);
		x=-dist*cos(radians);
		y=-dist*sin(radians);
		ob.id=id1;
		ob.x=x;
		ob.y=y;
		id1++;
		obs.push_back(ob);
	}
	
	for(int i=start;i<end;i++){
		if(scan.ranges[i]<scan.range_max){
			count=0;
			temp=i;
			
			while(scan.ranges[i]<scan.range_max){
				count++;
				i++;
				if(i==(end-1))
					break;
			}
			
			id=temp+count/2;
			dist=scan.ranges[id];
			deg=id/2; //not robust as considers size of range to be 720.
			dig.distance=dist;
			dig.degree=deg;
			div.push_back(dig);
			
			
			radians=deg*(3.142/180.0);
			x=-dist*cos(radians);
			y=-dist*sin(radians);
			ob.id=id1;
			ob.x=x;
			ob.y=y;
			id1++;
			obs.push_back(ob);
			
		}
	}
	
	
	runPf();
	
	PublishOdom();
	
	
}








int main(int argc, char** argv)
{
   
    ros::init(argc, argv, "lanmark_printer");


    ros::NodeHandle n;
    client = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state"); 
	
	client1 = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state"); 
	
	odom_pub = n.advertise<nav_msgs::Odometry>("/odom/Particle", 50);
	
	ros::Subscriber sub1 = n.subscribe("/cmd_vel", 10, get_speed_callback);
	
	ros::Subscriber sub = n.subscribe("/scan", 10, get_laser_callback);
	
	
	
	ros::spin();
	 
    return 0;
}

