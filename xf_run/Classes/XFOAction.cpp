#include "XFOAction.h"
#include "XFRunGameScene.h"
XFOAction::XFOAction(Node* node,XF* xf)
{
	this->node = node;
	this->xf = xf;
}

bool XFOAction::isDone() const{
     return !_target;
}
void XFOAction::step(float time){
	 if(_target)
	{
		_target->setPosition3D(_target->getPosition3D()+Vec3(0,0,100*time));
		if(_target->getPositionZ()>-40 && _target->getPositionZ()<10 )// enter the front
		{
			 Sprite3D * sprite = dynamic_cast<Sprite3D * >(_target);
			 auto dist =sprite->getPosition3D().distance(xf->getXF()->getPosition3D());
             if(dist<8)
             {
                 _target->setVisible(false);
                 _target->setScale(2);
                 _target->removeFromParent();
                 _target=nullptr;
				 auto a = (XFRunGameScene*)node;
				 a->gameover();
                 return ;
             }
        
		}
		if(_target->getPositionZ()>=35)
		{
			_target->removeFromParent();
			_target=nullptr;
		}
	}
}