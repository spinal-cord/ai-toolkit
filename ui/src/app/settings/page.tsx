'use client';

import { useEffect, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { deriveSamplePublicKey } from '@/utils/sampleKey';
import { deriveAuthPublicKey } from '@/utils/authKey';
import {
  generateConfigKeypair,
  storeConfigPrivateKey,
  clearConfigPrivateKey,
  hasConfigPrivateKey,
} from '@/utils/configKey';

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [debugStatus, setDebugStatus] = useState<'idle' | 'stopping' | 'success' | 'error'>('idle');
  const [debugMessage, setDebugMessage] = useState<string>('');
  const [authStatus, setAuthStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [authMessage, setAuthMessage] = useState<string>('');
  const [sampleStatus, setSampleStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [sampleMessage, setSampleMessage] = useState<string>('');
  const [datasetStatus, setDatasetStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [datasetMessage, setDatasetMessage] = useState<string>('');
  const [configGenStatus, setConfigGenStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [configGenMessage, setConfigGenMessage] = useState<string>('');
  const [generatedPrivKey, setGeneratedPrivKey] = useState<string>('');
  const [hasLocalPrivKey, setHasLocalPrivKey] = useState<boolean>(false);

  useEffect(() => {
    // Client-only: check whether this browser already holds a config private key.
    setHasLocalPrivKey(hasConfigPrivateKey());
  }, []);

  const handleGenerateConfigKeypair = async () => {
    setConfigGenStatus('saving');
    setConfigGenMessage('');
    try {
      // X25519 keypair generated entirely in the browser. The private key is
      // kept in localStorage (never sent to the server); the public key is
      // filled into the field below and persisted on "Save Settings".
      const { privateKeyB64, publicKeyB64 } = await generateConfigKeypair();
      storeConfigPrivateKey(privateKeyB64);
      setGeneratedPrivKey(privateKeyB64);
      setHasLocalPrivKey(true);
      setSettings(prev => ({ ...prev, CONFIG_PUBLIC_KEY: publicKeyB64 }));
      setConfigGenStatus('success');
      setConfigGenMessage(
        'Key pair generated. The private key is stored only in this browser (back it up below). Click "Save Settings" to store the public key on the server.',
      );
    } catch (error) {
      console.error('Error generating config key pair:', error);
      setConfigGenStatus('error');
      setConfigGenMessage('Failed to generate key pair. Your browser may not support X25519 WebCrypto.');
    } finally {
      setTimeout(() => setConfigGenStatus('idle'), 6000);
    }
  };

  const handleClearConfigEncryption = async () => {
    setConfigGenStatus('saving');
    setConfigGenMessage('');
    try {
      await apiClient.post('/api/settings', { ...settings, CLEAR_CONFIG_ENCRYPTION: true });
      clearConfigPrivateKey();
      setGeneratedPrivKey('');
      setHasLocalPrivKey(false);
      setSettings(prev => ({ ...prev, CONFIG_PUBLIC_KEY: '', CONFIG_ENCRYPTION_SET: false }));
      setConfigGenStatus('success');
      setConfigGenMessage('Config encryption removed - new configs are stored unencrypted.');
    } catch (error) {
      console.error('Error clearing config encryption:', error);
      setConfigGenStatus('error');
      setConfigGenMessage('Failed to remove config encryption.');
    } finally {
      setTimeout(() => setConfigGenStatus('idle'), 5000);
    }
  };

  const handleClearPassword = async () => {
    setAuthStatus('saving');
    setAuthMessage('');
    try {
      await apiClient.post('/api/settings', { ...settings, CLEAR_AUTH: true });
      setAuthStatus('success');
      setAuthMessage('Password removed - the UI is open again.');
      setSettings(prev => ({ ...prev, AUTH_PASSWORD_SET: false }));
    } catch (error) {
      console.error('Error clearing password:', error);
      setAuthStatus('error');
      setAuthMessage('Failed to remove password.');
    } finally {
      setTimeout(() => setAuthStatus('idle'), 5000);
    }
  };

  const handleClearSamplePassword = async () => {
    setSampleStatus('saving');
    setSampleMessage('');
    try {
      await apiClient.post('/api/settings', { ...settings, CLEAR_SAMPLE_PASSWORD: true });
      setSampleStatus('success');
      setSampleMessage('Sample password removed - new samples are stored unencrypted.');
      setSettings(prev => ({ ...prev, SAMPLE_PASSWORD_SET: false }));
    } catch (error) {
      console.error('Error clearing sample password:', error);
      setSampleStatus('error');
      setSampleMessage('Failed to remove sample password.');
    } finally {
      setTimeout(() => setSampleStatus('idle'), 5000);
    }
  };

  const handleClearDatasetPassword = async () => {
    setDatasetStatus('saving');
    setDatasetMessage('');
    try {
      await apiClient.post('/api/settings', { ...settings, CLEAR_DATASET_PASSWORD: true });
      setDatasetStatus('success');
      setDatasetMessage('Dataset password removed - datasets must be stored unencrypted.');
      setSettings(prev => ({ ...prev, DATASET_PASSWORD_SET: false }));
    } catch (error) {
      console.error('Error clearing dataset password:', error);
      setDatasetStatus('error');
      setDatasetMessage('Failed to remove dataset password.');
    } finally {
      setTimeout(() => setDatasetStatus('idle'), 5000);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    try {
      // If new passwords were entered, derive the public keys locally in the
      // browser and send ONLY the public keys to the server. The passwords
      // themselves never leave the browser.
      const payload: Record<string, unknown> = { ...settings };
      let authKeySet = false;
      let datasetKeySet = false;
      if (settings.AUTH_PASSWORD && settings.AUTH_PASSWORD.trim() !== '') {
        payload.AUTH_PUBLIC_KEY = await deriveAuthPublicKey(settings.AUTH_PASSWORD);
        authKeySet = true;
      }
      delete payload.AUTH_PASSWORD;
      if (settings.SAMPLE_PASSWORD && settings.SAMPLE_PASSWORD.trim() !== '') {
        payload.SAMPLE_PUBLIC_KEY = await deriveSamplePublicKey(settings.SAMPLE_PASSWORD);
      }
      delete payload.SAMPLE_PASSWORD;
      // The dataset password is symmetric: the plaintext is sent to the server
      // and passed to the training worker as the DATASET_PASSWORD env var.
      if (settings.DATASET_PASSWORD && settings.DATASET_PASSWORD.trim() !== '') {
        datasetKeySet = true;
      }

      await apiClient.post('/api/settings', payload);
      // clear the password fields: they are never sent back, blank = keep current
      setSettings(prev => ({
        ...prev,
        AUTH_PASSWORD: '',
        SAMPLE_PASSWORD: '',
        DATASET_PASSWORD: '',
        AUTH_PASSWORD_SET: authKeySet ? true : prev.AUTH_PASSWORD_SET,
        DATASET_PASSWORD_SET: datasetKeySet ? true : prev.DATASET_PASSWORD_SET,
      }));
      setStatus('success');
    } catch (error) {
      console.error('Error saving settings:', error);
      setStatus('error');
      if (error instanceof Error && /Ed25519|X25519|browser/i.test(error.message)) {
        // surface the "unsupported browser" message instead of a generic error
        alert(error.message);
      }
    } finally {
      setTimeout(() => setStatus('idle'), 2000);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  const handleStopAllJobs = async () => {
    setDebugStatus('stopping');
    setDebugMessage('');

    try {
      const response = await apiClient.post('/api/jobs/stop-all');
      const data = response.data;
      
      setDebugStatus('success');
      setDebugMessage(data.message);
    } catch (error: any) {
      console.error('Error stopping all running jobs:', error);
      setDebugStatus('error');
      setDebugMessage(error.response?.data?.error || 'Failed to stop running jobs');
    } finally {
      setTimeout(() => {
        setDebugStatus('idle');
        setDebugMessage('');
      }, 5000);
    }
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face Token
                    <div className="text-gray-500 text-sm ml-1">
                      Create a Read token on{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        {' '}
                        Huggingface
                      </a>{' '}
                      if you need to access gated/private models.
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter your Hugging Face token"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    Training Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      We will store your training information here. Must be an absolute path. If blank, it will default
                      to the output folder in the project root.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter training folder path"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    Dataset Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      Where we store and find your datasets.{' '}
                      <span className="text-orange-800">
                        Warning: This software may modify datasets so it is recommended you keep a backup somewhere else
                        or have a dedicated folder for this software.
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter datasets folder path"
                  />
                </div>
              </div>

              {/* Security */}
              <div>
                <h2 className="text-lg font-semibold mb-4">Security</h2>
                <div className="space-y-4">
                  <div>
                    <label htmlFor="AUTH_PASSWORD" className="block text-sm font-medium mb-2">
                      UI Password
                      <div className="text-gray-500 text-sm ml-1">
                        Protects the web UI. First launch works without a password; once set, the password is
                        required on every launch (dev and production alike). Only a public key derived from
                        it in your browser is stored on the server - the password itself is never transmitted
                        or stored.
                      </div>
                    </label>
                    <input
                      type="password"
                      id="AUTH_PASSWORD"
                      name="AUTH_PASSWORD"
                      value={settings.AUTH_PASSWORD}
                      onChange={handleChange}
                      className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                      placeholder={settings.AUTH_PASSWORD_SET ? 'Leave blank to keep current password' : 'Set a password (optional)'}
                      autoComplete="new-password"
                    />
                    {settings.AUTH_PASSWORD_SET && (
                      <button
                        type="button"
                        onClick={handleClearPassword}
                        disabled={authStatus === 'saving'}
                        className="mt-2 px-3 py-1.5 text-sm bg-red-900/60 hover:bg-red-800/60 border border-red-700 rounded-lg transition-colors disabled:opacity-50"
                      >
                        Remove password
                      </button>
                    )}
                  </div>

                  <div>
                    <label htmlFor="SAMPLE_PASSWORD" className="block text-sm font-medium mb-2">
                      Sample Encryption Password
                      <div className="text-gray-500 text-sm ml-1">
                        When set, all generated samples are encrypted at rest. Each sample keeps its
                        file extension (e.g. .png) but its contents are encrypted with a public key
                        derived from this password in your browser - only the public key is sent to
                        and stored on the server, the password itself is never transmitted. Decrypt
                        samples with: python scripts/decrypt_sample.py &lt;password&gt; &lt;sample&gt;
                        Applied to samples generated by jobs started after it is saved.
                      </div>
                    </label>
                    <input
                      type="password"
                      id="SAMPLE_PASSWORD"
                      name="SAMPLE_PASSWORD"
                      value={settings.SAMPLE_PASSWORD}
                      onChange={handleChange}
                      className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                      placeholder={settings.SAMPLE_PASSWORD_SET ? 'Leave blank to keep current password' : 'Set a sample encryption password (optional)'}
                      autoComplete="new-password"
                    />
                    {settings.SAMPLE_PASSWORD_SET && (
                      <button
                        type="button"
                        onClick={handleClearSamplePassword}
                        disabled={sampleStatus === 'saving'}
                        className="mt-2 px-3 py-1.5 text-sm bg-red-900/60 hover:bg-red-800/60 border border-red-700 rounded-lg transition-colors disabled:opacity-50"
                      >
                        Remove sample password
                      </button>
                    )}
                    {sampleStatus === 'success' && (
                      <div className="mt-2 text-sm text-green-500">{sampleMessage}</div>
                    )}
                    {sampleStatus === 'error' && (
                      <div className="mt-2 text-sm text-red-500">{sampleMessage}</div>
                    )}
                  </div>

                  <div>
                    <label htmlFor="DATASET_PASSWORD" className="block text-sm font-medium mb-2">
                      Dataset Password
                      <div className="text-gray-500 text-sm ml-1">
                        When set, dataset files (images, videos, audio, captions and the
                        latents/embeddings caches) can be encrypted at rest and are decrypted
                        on the fly into RAM while training. Unlike the sample password this one
                        is symmetric - it is stored on the server and passed to training jobs.
                        Encrypt/decrypt files with: python scripts/encrypt_dataset.py
                        &lt;password&gt; &lt;dataset_folder&gt; [--decrypt]. Applied to jobs started after
                        it is saved.
                      </div>
                    </label>
                    <input
                      type="password"
                      id="DATASET_PASSWORD"
                      name="DATASET_PASSWORD"
                      value={settings.DATASET_PASSWORD}
                      onChange={handleChange}
                      className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                      placeholder={settings.DATASET_PASSWORD_SET ? 'Leave blank to keep current password' : 'Set a dataset password (optional)'}
                      autoComplete="new-password"
                    />
                    {settings.DATASET_PASSWORD_SET && (
                      <button
                        type="button"
                        onClick={handleClearDatasetPassword}
                        disabled={datasetStatus === 'saving'}
                        className="mt-2 px-3 py-1.5 text-sm bg-red-900/60 hover:bg-red-800/60 border border-red-700 rounded-lg transition-colors disabled:opacity-50"
                      >
                        Remove dataset password
                      </button>
                    )}
                    {datasetStatus === 'success' && (
                      <div className="mt-2 text-sm text-green-500">{datasetMessage}</div>
                    )}
                    {datasetStatus === 'error' && (
                      <div className="mt-2 text-sm text-red-500">{datasetMessage}</div>
                    )}
                  </div>

                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <div>
                      <label htmlFor="TLS_CERT" className="block text-sm font-medium mb-2">
                        TLS Certificate Path
                        <div className="text-gray-500 text-sm ml-1">
                          Absolute path to the certificate file (e.g. /workspace/ssl/server.crt). Applied
                          automatically within ~30s - no restart needed.
                        </div>
                      </label>
                      <input
                        type="text"
                        id="TLS_CERT"
                        name="TLS_CERT"
                        value={settings.TLS_CERT}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="/workspace/ssl/server.crt"
                      />
                    </div>
                    <div>
                      <label htmlFor="TLS_KEY" className="block text-sm font-medium mb-2">
                        TLS Private Key Path
                        <div className="text-gray-500 text-sm ml-1">Absolute path to the private key file.</div>
                      </label>
                      <input
                        type="text"
                        id="TLS_KEY"
                        name="TLS_KEY"
                        value={settings.TLS_KEY}
                        onChange={handleChange}
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                        placeholder="/workspace/ssl/server.key"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Config Encryption */}
          <div>
            <h2 className="text-lg font-semibold mb-4">Config Encryption</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Job Config Encryption
                  <div className="text-gray-500 text-sm ml-1">
                    When enabled, job configs are encrypted with a key pair generated in this browser. Only this
                    browser (holding the private key) can decrypt a stored config - even the AI-toolkit cannot
                    decrypt it. The unencrypted config is still sent to start training.
                  </div>
                </label>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleGenerateConfigKeypair}
                    disabled={configGenStatus === 'saving'}
                    className="px-3 py-1.5 text-sm bg-blue-700 hover:bg-blue-600 border border-blue-600 rounded-lg transition-colors disabled:opacity-50"
                  >
                    Generate Key Pair
                  </button>
                  <span className="text-xs text-gray-400">
                    Public key on server: {settings.CONFIG_ENCRYPTION_SET ? 'set' : 'not set'} · Private key in
                    this browser: {hasLocalPrivKey ? 'yes' : 'no'}
                  </span>
                </div>
              </div>

              <div>
                <label htmlFor="CONFIG_PUBLIC_KEY" className="block text-sm font-medium mb-2">
                  Config Encryption Public Key
                  <div className="text-gray-500 text-sm ml-1">
                    The 32-byte X25519 public key (base64). Generated above, or paste a key you already have.
                    Saved to the server so this browser can encrypt configs.
                  </div>
                </label>
                <input
                  type="text"
                  id="CONFIG_PUBLIC_KEY"
                  name="CONFIG_PUBLIC_KEY"
                  value={settings.CONFIG_PUBLIC_KEY}
                  onChange={handleChange}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent font-mono text-xs"
                  placeholder="Paste or generate the public key (base64)"
                />
                {settings.CONFIG_ENCRYPTION_SET && (
                  <button
                    type="button"
                    onClick={handleClearConfigEncryption}
                    disabled={configGenStatus === 'saving'}
                    className="mt-2 px-3 py-1.5 text-sm bg-red-900/60 hover:bg-red-800/60 border border-red-700 rounded-lg transition-colors disabled:opacity-50"
                  >
                    Remove config encryption
                  </button>
                )}
              </div>

              {generatedPrivKey && (
                <div>
                  <label htmlFor="configPrivKey" className="block text-sm font-medium mb-2">
                    Private Key (this browser only - back it up)
                    <div className="text-gray-500 text-sm ml-1">
                      Anyone with this key can decrypt your stored configs. Store it somewhere safe. It is never
                      sent to the server.
                    </div>
                  </label>
                  <textarea
                    id="configPrivKey"
                    readOnly
                    value={generatedPrivKey}
                    rows={3}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg font-mono text-xs text-gray-300"
                  />
                  <button
                    type="button"
                    onClick={() => navigator.clipboard?.writeText(generatedPrivKey)}
                    className="mt-2 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-lg transition-colors"
                  >
                    Copy private key
                  </button>
                </div>
              )}

              {configGenMessage && (
                <div
                  className={`p-3 rounded-lg text-sm ${
                    configGenStatus === 'error'
                      ? 'bg-red-900/50 text-red-300 border border-red-700'
                      : configGenStatus === 'success'
                        ? 'bg-green-900/50 text-green-300 border border-green-700'
                        : 'bg-blue-900/40 text-blue-200 border border-blue-700'
                  }`}
                >
                  {configGenMessage}
                </div>
              )}
            </div>
          </div>

          <button
            type="submit"
            disabled={status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'saving' ? 'Saving...' : 'Save Settings'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
          {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
        </form>

        {/* Debug Section */}
        <div className="mt-12 pt-8 border-t border-gray-700">
          <h2 className="text-lg font-semibold text-red-400 mb-4">Debug Tools</h2>
          <div className="bg-gray-800 p-4 rounded-lg border border-red-600">
            <div className="flex flex-col space-y-4">
              <div>
                <h3 className="text-sm font-medium text-red-300 mb-2">Force Stop All Running Jobs</h3>
                <p className="text-gray-400 text-sm mb-3">
                  This will immediately change the status of all running jobs to "stopped" in the database. 
                  Use this only if jobs appear stuck in running state.
                </p>
                <button
                  type="button"
                  onClick={handleStopAllJobs}
                  disabled={debugStatus === 'stopping'}
                  className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {debugStatus === 'stopping' ? 'Stopping Jobs...' : 'Stop All Running Jobs'}
                </button>
              </div>

              {debugMessage && (
                <div className={`p-3 rounded-lg text-sm ${
                  debugStatus === 'success' 
                    ? 'bg-green-900 text-green-300 border border-green-700' 
                    : 'bg-red-900 text-red-300 border border-red-700'
                }`}>
                  {debugMessage}
                </div>
              )}
            </div>
          </div>
        </div>
      </MainContent>
    </>
  );
}
