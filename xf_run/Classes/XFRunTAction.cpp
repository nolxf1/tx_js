#include "XFRunTAction.h"
bool XFRunTAction::isDone() const{
     return !_target;
}
void XFRunTAction::step(float time){
	if(_target){
	   _target->setPosition3D (_target->getPosition3D()+Vec3(0,1*time,10*time));
	   if(_target->getPositionZ()>10){
		   CCLOG("remove");
		   _target->removeFromParent();
		   _target = nullptr;
	   }
	}
}