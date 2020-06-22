#include "XFSAction.h"
bool XFSAction::isDone() const
{
	//target执行动作的目标node
	return !_target;
}

void XFSAction::step(float time)
{
	if(_target)
	{
		_target->setPosition3D(_target->getPosition3D()+Vec3(0,0,100*time));
		if(_target->getPositionZ()>=200)
		{
			_target->removeFromParent(); 
			_target=nullptr;
		}
	} 
}