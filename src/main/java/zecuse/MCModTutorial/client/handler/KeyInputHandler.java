package zecuse.MCModTutorial.client.handler;

import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.InputEvent;
import zecuse.MCModTutorial.client.settings.KeyBinds;
import zecuse.MCModTutorial.reference.Key;
import zecuse.MCModTutorial.utility.LogHelper;

// This class handles what happens when the user presses the key associated with the defined key name.
public class KeyInputHandler
{
	// Return the name of the key pressed. It doesn't matter if it was remapped by the user.
	private static Key getPressedKey()
	{
		if(KeyBinds.charge.isPressed())
			return Key.CHARGE;
		else if(KeyBinds.release.isPressed())
			return Key.RELEASE;
		
		return Key.UNKNOWN;
	}
	
	// Listen for keyboard input. When it matches a defined key name, execute that event.
	@SubscribeEvent
	public void handleKeyInputEvent(InputEvent.KeyInputEvent event)
	{
		LogHelper.info(getPressedKey());
	}
}
