#include "XFCAction.h"
#include"XFRunGameScene.h"

XFCAction::XFCAction(Node* node,XF* xf)
{
	this->node = node;
	this->xf = xf;
	angle = 0;
}
bool XFCAction::isDone()const{
    return !_target;
}
void XFCAction::step(float time){
	if(_target){
	   _target->setRotation3D(Vec3(90,angle,180));
		angle+=time*140;
	   _target->setPosition3D (_target->getPosition3D()+Vec3(0,0,100*time));
	   if(_target->getPositionZ()>-40 && _target->getPositionZ()<10 ){//xf将要吃到金币
	       	Sprite3D * sprite = dynamic_cast<Sprite3D * >(_target);
            auto dist =sprite->getPosition3D().distance(xf->getXF()->getPosition3D());
            if(dist<7)
            {
				//播放捡到金币的声音
                auto a = (XFRunGameScene*)this->node;
                a->getCoin();
				_target->removeFromParent();
			    _target=nullptr;
                return ;
            }
	   }
	   if(_target->getPositionZ()>10){
	      _target->removeFromParent();
	      _target=nullptr;
	   }
	}
}
XFCAction::~XFCAction(void)
{
}
