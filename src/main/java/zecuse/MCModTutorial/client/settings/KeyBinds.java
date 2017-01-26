package zecuse.MCModTutorial.client.settings;

import org.lwjgl.input.Keyboard;
import net.minecraft.client.settings.KeyBinding;
import zecuse.MCModTutorial.reference.Names;

// This class registers a key name to a key in a category of Minecraft's controls gui.
// Minecraft handles when a user changes keybinds and they will stay to what the user set them.
public class KeyBinds
{
	public static KeyBinding charge = new KeyBinding(Names.Keys.CHARGE, Keyboard.KEY_C, Names.Keys.CATEGORY);
	public static KeyBinding release = new KeyBinding(Names.Keys.RELEASE, Keyboard.KEY_R, Names.Keys.CATEGORY);
}
