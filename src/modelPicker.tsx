/**
 * Model picker component for selecting AI provider and model.
 */

import * as React from 'react';
import { useState, useEffect } from 'react';
import { ReactWidget } from '@jupyterlab/apputils';
import { URLExt } from '@jupyterlab/coreutils';
import { ServerConnection } from '@jupyterlab/services';
import { IExtensionSettings } from './tokens';
import { SettingsManager } from './settings';

interface ModelInfo {
  id: string;
  name: string;
  date: string;
}

interface ModelsResponse {
  providers: Record<string, string>;
  models: Record<string, ModelInfo[]>;
}

interface ModelPickerProps {
  settings: IExtensionSettings & { set?: SettingsManager['set'] };
}

const PROVIDER_ORDER = ['anthropic', 'openai', 'gemini'];

function ModelPickerComponent({ settings }: ModelPickerProps): JSX.Element {
  const [providers, setProviders] = useState<Record<string, string>>({});
  const [modelsByProvider, setModelsByProvider] = useState<Record<string, ModelInfo[]>>({});
  const [selectedProvider, setSelectedProvider] = useState(settings.provider || 'anthropic');
  const [selectedModel, setSelectedModel] = useState(settings.defaultModel || '');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const serverSettings = ServerConnection.makeSettings();
        const url = URLExt.join(serverSettings.baseUrl, 'jupyvibe', 'models');
        
        const response = await ServerConnection.makeRequest(url, {}, serverSettings);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.status}`);
        }
        
        const data: ModelsResponse = await response.json();
        setProviders(data.providers);
        setModelsByProvider(data.models);
        
        // If current model isn't in the list for the provider, select first available
        const providerModels = data.models[settings.provider] || [];
        if (providerModels.length > 0 && !providerModels.find(m => m.id === settings.defaultModel)) {
          setSelectedModel(providerModels[0].id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load models');
      } finally {
        setLoading(false);
      }
    };
    
    fetchModels();
  }, [settings.provider, settings.defaultModel]);

  const handleProviderChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newProvider = e.target.value;
    setSelectedProvider(newProvider);
    
    // Select first model for new provider
    const providerModels = modelsByProvider[newProvider] || [];
    const newModel = providerModels.length > 0 ? providerModels[0].id : '';
    setSelectedModel(newModel);
    
    // Save to settings
    if (settings.set) {
      await settings.set('provider', newProvider);
      if (newModel) {
        await settings.set('defaultModel', newModel);
      }
    }
  };

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    
    // Save to settings
    if (settings.set) {
      await settings.set('defaultModel', newModel);
    }
  };

  if (loading) {
    return <span className="jupyvibe-model-picker-loading">Loading...</span>;
  }

  if (error) {
    return (
      <span className="jupyvibe-model-picker-error" title={error}>
        ⚠️ Error
      </span>
    );
  }

  const currentModels = modelsByProvider[selectedProvider] || [];

  return (
    <div className="jupyvibe-model-picker">
      <select
        className="jupyvibe-provider-select"
        value={selectedProvider}
        onChange={handleProviderChange}
        title="Select AI Provider"
      >
        {PROVIDER_ORDER.filter(p => p in providers).map(providerId => (
          <option key={providerId} value={providerId}>
            {providers[providerId]}
          </option>
        ))}
      </select>
      <select
        className="jupyvibe-model-select"
        value={selectedModel}
        onChange={handleModelChange}
        title="Select Model"
      >
        {currentModels.map(model => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
    </div>
  );
}

export class ModelPickerWidget extends ReactWidget {
  private _settings: IExtensionSettings & { set?: SettingsManager['set'] };

  constructor(settings: IExtensionSettings) {
    super();
    this._settings = settings as IExtensionSettings & { set?: SettingsManager['set'] };
    this.addClass('jupyvibe-model-picker-widget');
  }

  render(): JSX.Element {
    return <ModelPickerComponent settings={this._settings} />;
  }
}
