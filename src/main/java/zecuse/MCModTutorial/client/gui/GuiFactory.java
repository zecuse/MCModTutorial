package zecuse.MCModTutorial.client.gui;

import java.util.Set;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiScreen;
import net.minecraftforge.fml.client.IModGuiFactory;

// This class runs the setup for the configuration gui.
public class GuiFactory implements IModGuiFactory
{

	@Override
	public void initialize(Minecraft minecraftInstance){}

	// This method runs the mod's config gui.
	@Override
	public Class<? extends GuiScreen> mainConfigGuiClass()
	{
		return ModGuiConfig.class;
	}

	@Override
	public Set<RuntimeOptionCategoryElement> runtimeGuiCategories(){return null;}

	@SuppressWarnings("deprecation")
	@Override
	public RuntimeOptionGuiHandler getHandlerFor(RuntimeOptionCategoryElement element){return null;}
}
