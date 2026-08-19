'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface Settings {
  HF_TOKEN: string;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  // Security
  AUTH_PASSWORD: string; // new password to set (blank = keep current); never returned by the API
  TLS_CERT: string; // absolute path to TLS certificate file
  TLS_KEY: string; // absolute path to TLS private key file
  AUTH_PASSWORD_SET: boolean; // whether a password is currently configured
  SAMPLE_PASSWORD: string; // sample-encryption password (blank = keep current); never returned by the API
  SAMPLE_PASSWORD_SET: boolean; // whether a sample-encryption password is currently configured
  DATASET_PASSWORD: string; // dataset password (blank = keep current); never returned by the API
  DATASET_PASSWORD_SET: boolean; // whether a dataset password is currently configured
  // Config encryption (X25519 keypair generated in the browser)
  CONFIG_PUBLIC_KEY: string; // public key (base64) used to encrypt job configs; safe to expose
  CONFIG_ENCRYPTION_SET: boolean; // whether a config-encryption public key is configured
}

export default function useSettings() {
  const [settings, setSettings] = useState({
    HF_TOKEN: '',
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    AUTH_PASSWORD: '',
    TLS_CERT: '',
    TLS_KEY: '',
    AUTH_PASSWORD_SET: false,
    SAMPLE_PASSWORD: '',
    SAMPLE_PASSWORD_SET: false,
    DATASET_PASSWORD: '',
    DATASET_PASSWORD_SET: false,
    CONFIG_PUBLIC_KEY: '',
    CONFIG_ENCRYPTION_SET: false,
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        console.log('Settings:', data);
        setSettings({
          HF_TOKEN: data.HF_TOKEN || '',
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
          AUTH_PASSWORD: '',
          TLS_CERT: data.TLS_CERT || '',
          TLS_KEY: data.TLS_KEY || '',
          AUTH_PASSWORD_SET: Boolean(data.AUTH_PASSWORD_SET),
          SAMPLE_PASSWORD: '',
          SAMPLE_PASSWORD_SET: Boolean(data.SAMPLE_PASSWORD_SET),
          DATASET_PASSWORD: '',
          DATASET_PASSWORD_SET: Boolean(data.DATASET_PASSWORD_SET),
          CONFIG_PUBLIC_KEY: data.CONFIG_PUBLIC_KEY || '',
          CONFIG_ENCRYPTION_SET: Boolean(data.CONFIG_ENCRYPTION_SET),
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
